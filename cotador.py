from __future__ import annotations
import os
import re
import json
import math
import unicodedata
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rapidfuzz import process, fuzz

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================================================
# CONFIGURAÇÃO
# =========================================================
APP_TITLE = "PMG Cotador IA - V8.6 Universal"
ARQUIVO_CLIENTES = "clientes_salvos.json"
ARQUIVO_APRENDIZADO = "aprendizado_cotacao.json"

# COLE SUA CHAVE AQUI, SE QUISER USAR OPENAI DIRETO NO CÓDIGO.
# Exemplo: OPENAI_API_KEY = ""
OPENAI_API_KEY = ""

OPENAI_MODEL = "gpt-4.1-mini"

DESCONTO_PADRAO = 2.95
MAX_OPCOES_LISTA = 20

tabela: Optional[pd.DataFrame] = None
catalogo: Optional[pd.DataFrame] = None

ultimo_texto_cliente = ""
ultimo_texto_interno = ""


# =========================================================
# UTIL
# =========================================================
def caminho_local(nome: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), nome)


def normalizar(txt: Any) -> str:
    txt = str(txt).lower().strip()
    txt = unicodedata.normalize("NFD", txt)
    txt = "".join(c for c in txt if unicodedata.category(c) != "Mn")
    txt = txt.replace("ç", "c")
    txt = re.sub(r"[^a-z0-9 ]", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def dinheiro(v: Any) -> str:
    try:
        return f"R$ {float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "R$ 0,00"


def limpar_preco(v: Any) -> float:
    s = str(v).replace("R$", "").replace(" ", "").strip()
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0


def fmt_num(v: Any) -> str:
    try:
        n = float(v)
        if n.is_integer():
            return str(int(n))
        return f"{n:.2f}".replace(".", ",")
    except Exception:
        return str(v)


def obter_openai_key() -> str:
    return (OPENAI_API_KEY or os.getenv("OPENAI_API_KEY") or "").strip()


def tem_openai() -> bool:
    return bool(obter_openai_key()) and OpenAI is not None


# =========================================================
# SINÔNIMOS E REGRAS DE PRODUTO
# =========================================================
SINONIMOS = {
    "mucarela": ["mucarela", "muçarela", "mussarela", "mozarela", "muzzarela", "muca", "mussarela"],
    "requeijao": ["requeijao", "requeijão", "catupiry", "requijao", "req"],
    "calabresa": ["calabresa", "calabreza", "calaresa", "calaresas", "calaresa", "linguica calabresa", "linguiça calabresa"],
    "presunto": ["presunto"],
    "apresuntado": ["apresuntado"],
    "manteiga": ["manteiga"],
    "margarina": ["margarina"],
    "farinha": ["farinha", "farinha pizza", "farinha de trigo"],
    "molho tomate": ["molho tomate", "molho de tomate", "tomate molho"],
    "tomate pelado": ["tomate pelado"],
    "pepperoni": ["pepperoni"],
    "bacon": ["bacon"],
    "batata": ["batata", "batatas", "batat", "batata palito", "batata congelada", "batata pre frita", "batata pré frita"],
    "azeitona": ["azeitona", "azeitonas"],
    "alho": ["alho"],
    "parmesao": ["parmesao", "parmesão"],
    "provolone": ["provolone"],
    "gorgonzola": ["gorgonzola"],
    "cream cheese": ["cream cheese", "creme cheese"],
    "cheddar": ["cheddar"],
    "oleo": ["oleo", "óleo", "oleo soja", "óleo soja"],
    "coca cola": ["coca", "coca cola", "coca-cola"],
    "atum": ["atum"],
    "salame": ["salame"],
}

EXCLUSOES_PADRAO = {
    "mucarela": ["bufala", "bocconcino", "bolinha", "cereja", "cerejas", "zero lactose", "cobertura"],
    "presunto": ["apresuntado"],
}

MARCAS_COMPOSTAS = [
    "tres marias", "três marias", "sabor de minas", "monte castelo",
    "arco bello", "gomes da costa", "bella italia", "sao vicente",
    "frigo nosso", "boi brasil", "da vaca", "alto do vale",
]

PALAVRAS_MEDIDAS = {
    "kg", "g", "grama", "gramas", "kilo", "quilo", "quilos",
    "cx", "caixa", "caixas",
    "pct", "pacote", "pacotes",
    "fdo", "fd", "frd", "fardo", "fardos",
    "bis", "bisnaga", "bisnagas",
    "bag", "bags",
    "bd", "balde", "baldes",
    "un", "und", "unidade", "unidades",
    "pc", "pç", "peca", "peça", "pecas", "peças",
    "gl", "galao", "galão", "galoes", "galões",
    "lt", "lata", "latas",
    "vd", "vidro", "vidros",
    "l", "litro", "litros", "ml",
}


def singular_unidade(u: str) -> str:
    u = normalizar(u)
    mapa = {
        "kg": "kg", "kilo": "kg", "quilo": "kg", "quilos": "kg",
        "g": "g", "grama": "g", "gramas": "g",
        "l": "l", "litro": "l", "litros": "l",
        "ml": "ml",
        "cx": "caixa", "caixa": "caixa", "caixas": "caixa",
        "pct": "pacote", "pacote": "pacote", "pacotes": "pacote",
        "fdo": "fardo", "fd": "fardo", "frd": "fardo", "fardo": "fardo", "fardos": "fardo",
        "bis": "bisnaga", "bisnaga": "bisnaga", "bisnagas": "bisnaga",
        "bag": "bag", "bags": "bag",
        "bd": "balde", "balde": "balde", "baldes": "balde",
        "un": "unidade", "und": "unidade", "unidade": "unidade", "unidades": "unidade",
        "pc": "peça", "pç": "peça", "peca": "peça", "peça": "peça", "pecas": "peça", "peças": "peça",
        "gl": "galão", "galao": "galão", "galão": "galão", "galoes": "galão", "galões": "galão",
        "lt": "lata", "lata": "lata", "latas": "lata",
        "vd": "vidro", "vidro": "vidro", "vidros": "vidro",
    }
    return mapa.get(u, u or "unidade")


def nome_unidade_cliente(u: str) -> str:
    u = singular_unidade(u)
    mapa = {
        "kg": "kg",
        "caixa": "caixa",
        "pacote": "pacote",
        "fardo": "fardo",
        "bisnaga": "bisnaga",
        "bag": "bag",
        "balde": "balde",
        "unidade": "unidade",
        "peça": "peça",
        "galão": "galão",
        "lata": "lata",
        "vidro": "vidro",
        "l": "litro",
    }
    return mapa.get(u, u)


def plural_unidade(u: str) -> str:
    u = nome_unidade_cliente(u)
    if u == "kg":
        return "kg"
    if u.endswith("ão"):
        return u[:-2] + "ões"
    if u.endswith("l"):
        return u + "s"
    return u + "s"


def detectar_produto_base(texto: str) -> Optional[str]:
    n = corrigir_busca(texto) if "corrigir_busca" in globals() else normalizar(texto)

    # Presunto e apresuntado são diferentes.
    if "apresuntado" in n:
        return "apresuntado"
    if re.search(r"\bpresunto\b", n):
        return "presunto"

    for base, termos in SINONIMOS.items():
        for t in termos:
            tn = normalizar(t)
            if re.search(rf"\b{re.escape(tn)}\b", n):
                return base

    return None


def corrigir_busca(texto: str) -> str:
    n = normalizar(texto)

    trocas_frases = {
        "mais baratas": "mais barato",
        "mais baratos": "mais barato",
        "mais barata": "mais barato",
        "masi bartata": "mais barato",
        "masi barata": "mais barato",
        "masi barato": "mais barato",
        "mais bartata": "mais barato",
        "menor preco": "mais barato",
        "menores precos": "mais barato",
    }

    for a, b in trocas_frases.items():
        n = n.replace(normalizar(a), normalizar(b))

    trocas_palavras = {
        "muçarela": "mucarela",
        "mussarela": "mucarela",
        "mozarela": "mucarela",
        "muzzarela": "mucarela",
        "mucarelas": "mucarela",
        "mussarelas": "mucarela",
        "muçarelas": "mucarela",
        "mozarelas": "mucarela",
        "calabreza": "calabresa",
        "calaresa": "calabresa",
        "calaresa": "calabresa",
        "calaresas": "calabresa",
        "calabresas": "calabresa",
        "requeijoes": "requeijao",
        "requeijões": "requeijao",
        "requijao": "requeijao",
        "requijoes": "requeijao",
        "presuntos": "presunto",
        "manteigas": "manteiga",
        "azeitonas": "azeitona",
        "gorgonzolas": "gorgonzola",
        "catupiri": "catupiry",
        "cheddars": "cheddar",
        "farinhas": "farinha",
        "bacons": "bacon",
        "pepperonis": "pepperoni",
        "batat": "batata",
        "batatas": "batata",
    }

    tokens = []
    for t in n.split():
        tokens.append(trocas_palavras.get(t, t))

    return " ".join(tokens).strip()


def tem_modo_mais_barato(texto: str) -> bool:
    n = corrigir_busca(texto)
    return (
        "mais barato" in n
        or "menor preco" in n
        or "menores precos" in n
        or fuzz.partial_ratio("mais barato", n) >= 82
    )



NUMEROS_PT = {
    "um": 1, "uma": 1,
    "dois": 2, "duas": 2,
    "tres": 3, "três": 3,
    "quatro": 4,
    "cinco": 5,
    "seis": 6,
    "sete": 7,
    "oito": 8,
    "nove": 9,
    "dez": 10,
    "onze": 11,
    "doze": 12,
    "treze": 13,
    "quatorze": 14, "catorze": 14,
    "quinze": 15,
    "dezesseis": 16, "dezaseis": 16,
    "dezessete": 17, "dezasete": 17,
    "dezoito": 18,
    "dezenove": 19,
    "vinte": 20,
}


def normalizar_comando_inicial_quantidade(linha: str) -> str:
    """
    Ajusta frases naturais antes da interpretação.
    Ex:
    - "me traga cinco mussarelas mais baratas" -> "5 mussarelas mais baratas"
    - "traga 20 requeijões mais baratos" -> "20 requeijões mais baratos"
    - "quero oito atuns mais baratos" -> "8 atuns mais baratos"
    """
    s = str(linha or "").strip()

    # remove comandos comuns no começo da frase
    s = re.sub(
        r"^\s*(por\s+favor\s+)?(me\s+)?(traga|traz|mande|manda|quero|queria|lista|liste|mostre|busque|buscar|pesquise|pesquisar)\s+",
        "",
        s,
        flags=re.I,
    ).strip()

    # remove artigos soltos após o comando
    s = re.sub(r"^\s*(os|as|o|a)\s+", "", s, flags=re.I).strip()

    # transforma número por extenso no começo em dígito
    m = re.match(r"^([A-Za-zÀ-ÿ]+)\b(.*)$", s)
    if m:
        palavra = normalizar(m.group(1))
        resto = m.group(2) or ""
        if palavra in NUMEROS_PT:
            s = f"{NUMEROS_PT[palavra]}{resto}"

    return s.strip()


def detectar_marcas(texto: str) -> List[str]:
    n = normalizar(texto)
    marcas = []

    # compostas primeiro
    for m in MARCAS_COMPOSTAS:
        mn = normalizar(m)
        if mn in n:
            marcas.append(mn)

    # marcas comuns aparecem como palavras soltas
    palavras_ignorar = set(PALAVRAS_MEDIDAS) | {
        "quero", "traga", "mande", "me", "mais", "barato", "barata", "baratos", "baratas",
        "com", "sem", "amido", "desconto", "preco", "preço", "por", "kg", "lista", "total",
        "nao", "não", "de", "da", "do", "e", "ou", "os", "as", "o", "a", "tipos", "marcas",
    }

    marcas_conhecidas = [
        "scala", "tirolez", "catupiry", "aurora", "frimesa", "sadia", "seara", "perdigao",
        "perdigão", "italac", "camil", "ekma", "hellmanns", "heinz", "anaconda", "nita",
        "tres", "três", "marias", "hm", "realac", "bonissimo", "boníssimo", "tradicao",
        "tradição", "deale", "natville", "domilac", "piloto", "crioulo", "lira", "apolo",
        "pompeia", "pompeia", "scalon", "clara", "milk", "rekeminas", "cremille",
        "affamato", "vigor", "coamo", "liza", "soya", "rezende", "pamplona", "lactofrios",
        "ceratti", "rjr", "arco", "bello", "predilecta", "ole", "olé"
    ]

    for m in marcas_conhecidas:
        mn = normalizar(m)
        if re.search(rf"\b{re.escape(mn)}\b", n):
            marcas.append(mn)

    # junta tres + marias
    if "tres" in marcas and "marias" in marcas:
        marcas = [x for x in marcas if x not in ["tres", "marias"]]
        marcas.append("tres marias")

    return sorted(list(set(marcas)))


# =========================================================
# PLANILHA E CATÁLOGO INTELIGENTE
# =========================================================
def padronizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    mapa = {}
    for col in df.columns:
        c = normalizar(col)
        if c in ["cod", "codigo", "id"]:
            mapa[col] = "COD"
        elif c in ["produto", "produtos", "descricao", "nome"]:
            mapa[col] = "PRODUTO"
        elif c in ["vend por", "vendpor", "vendido por", "unidade", "un", "und"]:
            mapa[col] = "VEND_POR"
        elif c in ["preco", "preco 0", "preco0", "valor"]:
            mapa[col] = "PRECO"

    df = df.rename(columns=mapa)

    faltando = [c for c in ["PRODUTO", "VEND_POR", "PRECO"] if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas faltando: {faltando}. Use COD | PRODUTOS | VEND. POR | PREÇO")

    if "COD" not in df.columns:
        df["COD"] = ""

    df["COD"] = df["COD"].astype(str).str.strip()
    df["PRODUTO"] = df["PRODUTO"].astype(str).str.replace("\n", " ", regex=False).str.strip()
    df["VEND_POR"] = df["VEND_POR"].astype(str).str.strip().str.upper()
    df["PRECO"] = df["PRECO"].apply(limpar_preco)
    df["NORM"] = df["PRODUTO"].apply(normalizar)

    df = df[df["PRODUTO"].str.len() > 1].copy()
    df = df[df["PRECO"] > 0].copy()
    return df.reset_index(drop=True)



def carregar_csv_ou_excel(arquivo: str) -> pd.DataFrame:
    """
    Lê CSV, Excel ou PDF da tabela PMG.

    PDF PMG:
    O PDF quebra alguns produtos em 2 ou 3 linhas, por exemplo:

        BATATA PALITO CONGELADA PRÉ FRITA 9 MM...
        9086 CX R$ 2 02,71
        STEALTH FRIES ... (CX 6 PCT)

    Esse parser junta:
    - linhas antes do código
    - linha do código + unidade + preço
    - continuações depois do preço

    Resultado final:
    COD | PRODUTO completo | VEND_POR | PRECO
    """
    arquivo_lower = arquivo.lower()

    if arquivo_lower.endswith(".pdf"):
        try:
            import pdfplumber
        except Exception:
            raise ValueError(
                "A biblioteca pdfplumber não está instalada. Rode: python -m pip install pdfplumber"
            )

        unidades_validas = {
            "KG", "G", "PCT", "CX", "FD", "FDO", "UN", "UND", "PÇ", "PC",
            "BD", "BIS", "BAG", "GL", "LT", "VD", "FR", "SC", "BARR", "L"
        }

        def limpar_linha_pdf(linha: str) -> str:
            linha = re.sub(r"\s+", " ", str(linha or "")).strip()
            return linha

        def limpar_preco_pdf(preco: str) -> str:
            # PDF às vezes vem assim:
            # 2 02,71 -> 202,71
            # 9 3,55  -> 93,55
            # 5 ,54   -> 5,54
            preco = str(preco or "").strip()
            preco = re.sub(r"\s+", "", preco)
            preco = preco.replace("R$", "").strip()
            return preco

        def eh_ruido(linha: str) -> bool:
            upper = linha.upper()
            return (
                not linha
                or upper.startswith("COD PRODUTOS")
                or upper.startswith("COD ")
                or "TABELA DE PRODUTOS" in upper
                or "ENVIE SUA LISTA" in upper
                or "FRETE GRÁTIS" in upper
                or "FRETE GRATIS" in upper
                or "MAIS DE 1.900" in upper
            )

        # Linha completa: 9056 PRODUTO ... LT R$ 15,19
        padrao_linha_completa = re.compile(
            r"^(\d{1,6})\s+(.+?)\s+([A-ZÇÃÕÁÉÍÓÚÂÊÔÜ]{1,10})\s+R\$\s*([0-9\s.,]+)$",
            flags=re.I
        )

        # Linha âncora quebrada: 9086 CX R$ 2 02,71
        padrao_ancora = re.compile(
            r"^(\d{1,6})\s+([A-ZÇÃÕÁÉÍÓÚÂÊÔÜ]{1,10})\s+R\$\s*([0-9\s.,]+)$",
            flags=re.I
        )

        registros: List[Dict[str, str]] = []

        with pdfplumber.open(arquivo) as pdf:
            for page in pdf.pages:
                texto = page.extract_text(x_tolerance=1, y_tolerance=3)

                if not texto:
                    continue

                linhas = [
                    limpar_linha_pdf(l)
                    for l in texto.split("\n")
                ]

                linhas = [
                    l for l in linhas
                    if not eh_ruido(l)
                ]

                pendentes: List[str] = []
                i = 0

                while i < len(linhas):
                    linha = linhas[i]

                    # 1) Caso simples: começa com código e já termina com unidade + preço.
                    m_completa = padrao_linha_completa.match(linha)
                    if m_completa:
                        cod = m_completa.group(1).strip()
                        produto = m_completa.group(2).strip()
                        vend_por = m_completa.group(3).strip().upper()
                        preco = limpar_preco_pdf(m_completa.group(4))

                        if vend_por in unidades_validas and produto.upper() not in ["PRODUTOS", "PREÇO", "PRECO"]:
                            registros.append({
                                "COD": cod,
                                "PRODUTO": produto,
                                "VEND_POR": vend_por,
                                "PRECO": preco,
                            })

                        pendentes = []
                        i += 1
                        continue

                    # 2) Caso quebrado: produto veio antes, código/unidade/preço no meio.
                    m_ancora = padrao_ancora.match(linha)
                    if m_ancora:
                        cod = m_ancora.group(1).strip()
                        vend_por = m_ancora.group(2).strip().upper()
                        preco = limpar_preco_pdf(m_ancora.group(3))

                        partes_produto = [p for p in pendentes if p]
                        pendentes = []

                        # Continuações depois do preço pertencem ao produto atual
                        # até fechar embalagem com ")".
                        j = i + 1
                        while j < len(linhas):
                            prox = linhas[j]

                            # Se a próxima já é uma nova linha completa, não pertence ao produto atual.
                            if padrao_linha_completa.match(prox) or padrao_ancora.match(prox):
                                break

                            partes_produto.append(prox)

                            produto_tem_embalagem_fechada = ")" in " ".join(partes_produto)
                            if produto_tem_embalagem_fechada:
                                j += 1
                                break

                            j += 1

                        produto = " ".join(partes_produto)
                        produto = re.sub(r"\s+", " ", produto).strip()

                        if vend_por in unidades_validas and produto and produto.upper() not in ["PRODUTOS", "PREÇO", "PRECO"]:
                            registros.append({
                                "COD": cod,
                                "PRODUTO": produto,
                                "VEND_POR": vend_por,
                                "PRECO": preco,
                            })

                        i = j
                        continue

                    # 3) Linha de produto antes da âncora.
                    # Ex: BATATA PALITO CONGELADA...
                    pendentes.append(linha)

                    # Evita acumular lixo demais se o PDF tiver bloco estranho.
                    if len(pendentes) > 6:
                        pendentes = pendentes[-6:]

                    i += 1

        if not registros:
            raise ValueError(
                "Não consegui extrair produtos do PDF. Verifique se o PDF tem texto selecionável ou se é imagem escaneada."
            )

        df_pdf = pd.DataFrame(registros)

        # Remove duplicados causados por leitura de página.
        df_pdf = df_pdf.drop_duplicates(subset=["COD", "PRODUTO", "VEND_POR", "PRECO"])

        return df_pdf.reset_index(drop=True)

    if arquivo_lower.endswith(".csv"):
        ultimo_erro = None

        for enc in ["utf-8-sig", "latin1", "cp1252", "iso-8859-1"]:
            for sep in [None, ";", ",", "\t"]:
                try:
                    df = pd.read_csv(
                        arquivo,
                        sep=sep,
                        engine="python",
                        encoding=enc,
                        dtype=str
                    )

                    if len(df.columns) >= 3:
                        return df

                except Exception as e:
                    ultimo_erro = e

        raise ValueError(f"Não consegui ler CSV. Último erro: {ultimo_erro}")

    return pd.read_excel(arquivo, dtype=str)


def parse_peso_unidade(produto: str) -> Dict[str, Any]:
    base = produto.split("(")[0]
    matches = re.findall(r"(\d+(?:[,.]\d+)?)\s*(KG|G|ML|L)\b", base, flags=re.I)
    if not matches:
        return {"qtd": None, "un": None, "kg": None, "litros": None}

    valor, un = matches[-1]
    valor = float(valor.replace(",", "."))
    un = un.upper()

    kg = None
    litros = None
    if un == "KG":
        kg = valor
    elif un == "G":
        kg = valor / 1000
    elif un == "L":
        litros = valor
    elif un == "ML":
        litros = valor / 1000

    return {"qtd": valor, "un": un, "kg": kg, "litros": litros}


def parse_embalagem(produto: str) -> Dict[str, Any]:
    res = {"container": None, "qtd": None, "un_interna": None, "texto": None}

    m = re.search(r"\(([^)]*)\)", produto)
    if not m:
        return res

    texto = m.group(1).replace('"', '').strip().upper()
    res["texto"] = texto

    # Ex: CX 6 PÇ / FDO 25 KG / CX 12 BIS / PCT 12 LT
    mm = re.search(r"\b(CX|FDO|FD|PCT|BD|LT|GL)\s*(\d+(?:[,.]\d+)?)?\s*(KG|G|PÇ|PC|PCT|UN|UND|BIS|BAG|BD|GL|LT|L|VD|FR)?\b", texto, flags=re.I)
    if mm:
        container = singular_unidade(mm.group(1))
        qtd = float(mm.group(2).replace(",", ".")) if mm.group(2) else None
        un_interna = singular_unidade(mm.group(3)) if mm.group(3) else None
        res.update({"container": container, "qtd": qtd, "un_interna": un_interna})

    return res


def calcular_precos(produto: str, vend_por: str, preco: float) -> Dict[str, Any]:
    vend = singular_unidade(vend_por)
    peso = parse_peso_unidade(produto)
    emb = parse_embalagem(produto)

    kg_item = peso["kg"]
    litros_item = peso["litros"]

    preco_kg = None
    preco_litro = None
    preco_unidade = None
    preco_caixa = None

    # Regra principal:
    # Se vendido por KG, PRECO é preço/kg.
    # Se vendido por CX/FDO, PRECO é preço do container.
    # Se vendido por BIS/PCT/UN/PÇ/BAG, PRECO é preço da unidade interna.
    if vend == "kg":
        preco_kg = preco
        if kg_item:
            preco_unidade = preco * kg_item
    elif vend == "l":
        preco_litro = preco
        if litros_item:
            preco_unidade = preco * litros_item
    elif vend in ["caixa", "fardo"]:
        preco_caixa = preco
        if emb["qtd"]:
            if emb["un_interna"] == "kg":
                preco_kg = preco / emb["qtd"]
            elif kg_item:
                # caixa com X peças de Y kg
                preco_unidade = preco / emb["qtd"]
                preco_kg = preco_unidade / kg_item
            else:
                preco_unidade = preco / emb["qtd"]
    else:
        preco_unidade = preco
        if kg_item:
            preco_kg = preco / kg_item
        if litros_item:
            preco_litro = preco / litros_item

    # Preço da caixa/fardo, sempre que possível.
    if preco_caixa is None and emb["qtd"]:
        if emb["un_interna"] == "kg" and preco_kg is not None:
            preco_caixa = preco_kg * emb["qtd"]
        elif emb["un_interna"] == "l" and preco_litro is not None:
            preco_caixa = preco_litro * emb["qtd"]
        elif preco_unidade is not None:
            preco_caixa = preco_unidade * emb["qtd"]
        elif vend == "kg" and kg_item:
            preco_caixa = preco * kg_item * emb["qtd"]

    return {
        "vend": vend,
        "peso": peso,
        "emb": emb,
        "preco_tabela": preco,
        "preco_kg": preco_kg,
        "preco_litro": preco_litro,
        "preco_unidade": preco_unidade,
        "preco_caixa": preco_caixa,
    }


def identificar_marca(produto: str, produto_base: Optional[str]) -> str:
    n = normalizar(produto)

    marcas = detectar_marcas(produto)
    if marcas:
        # remove termos que são produto base, deixa marca mais provável
        marcas = [m for m in marcas if m not in ["mucarela", "requeijao", "calabresa", "presunto"]]
        if "tres marias" in marcas:
            return "TRÊS MARIAS"
        if marcas:
            return marcas[0].upper()

    # fallback: tenta palavra após produto_base
    if produto_base:
        nb = normalizar(produto_base)
        partes = n.split()
        if nb in n:
            idx = n.find(nb)
            resto = n[idx + len(nb):].strip().split()
            if resto:
                cand = resto[0]
                if cand not in PALAVRAS_MEDIDAS:
                    return cand.upper()

    return ""


def enriquecer_catalogo(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for _, row in df.iterrows():
        nome = row["PRODUTO"]
        base = detectar_produto_base(nome)
        precos = calcular_precos(nome, row["VEND_POR"], float(row["PRECO"]))

        marca = identificar_marca(nome, base)

        linha = dict(row)
        linha["BASE"] = base or ""
        linha["MARCA"] = marca
        linha["VEND"] = precos["vend"]
        linha["PESO_KG_ITEM"] = precos["peso"]["kg"]
        linha["PESO_L_ITEM"] = precos["peso"]["litros"]
        linha["EMB_CONTAINER"] = precos["emb"]["container"]
        linha["EMB_QTD"] = precos["emb"]["qtd"]
        linha["EMB_UN_INTERNA"] = precos["emb"]["un_interna"]
        linha["PRECO_KG"] = precos["preco_kg"]
        linha["PRECO_LITRO"] = precos["preco_litro"]
        linha["PRECO_UNIDADE"] = precos["preco_unidade"]
        linha["PRECO_CAIXA"] = precos["preco_caixa"]
        linhas.append(linha)

    return pd.DataFrame(linhas)


def termos_importantes_catalogo() -> List[str]:
    """
    Cria um vocabulário automaticamente a partir da planilha.
    Isso ajuda a corrigir erro de digitação para qualquer produto/marca,
    não só pizzaria.
    """
    if catalogo is None:
        return []

    stop = set(PALAVRAS_MEDIDAS) | {
        "com", "sem", "de", "da", "do", "para", "tipo", "grande", "pequeno", "pequena",
        "medio", "media", "médio", "média", "congelado", "congelada", "resfriado",
        "resfriada", "tradicional", "food", "service", "em", "ao", "a", "o", "e",
        "cx", "fdo", "pct", "kg", "un", "pc", "pç"
    }

    termos = set()

    for nome in catalogo["NORM"].astype(str).tolist():
        for token in nome.split():
            if len(token) >= 4 and token not in stop and not token.isdigit():
                termos.add(token)

    # inclui bases conhecidas e marcas extraídas
    for base in SINONIMOS.keys():
        termos.add(normalizar(base))
    for marca in catalogo["MARCA"].dropna().astype(str).tolist():
        mn = normalizar(marca)
        if len(mn) >= 2:
            termos.add(mn)

    return sorted(termos)


def corrigir_termo_por_catalogo(token: str) -> str:
    """
    Corrige uma palavra digitada errada usando vocabulário da própria planilha.
    Ex: calaresa -> calabresa, prezunto -> presunto, gorgonzla -> gorgonzola.
    """
    t = normalizar(token)

    # Palavras de comando/ordenação NÃO podem ser corrigidas pelo catálogo.
    # Sem isso, "mais barata" pode virar "mais barao" se existir marca/produto BARÃO.
    palavras_protegidas = {
        "mais", "barato", "barata", "baratos", "baratas",
        "menor", "preco", "precos", "desconto", "lista",
        "quero", "mande", "manda", "traga", "traz", "pesquisa", "pesquisar",
        "busca", "buscar", "cotacao", "orcamento"
    }

    if t in palavras_protegidas:
        return t

    if len(t) < 4 or t in PALAVRAS_MEDIDAS:
        return t

    # Correções manuais fortes primeiro
    mapa_forte = {
        "calaresa": "calabresa",
        "calaresas": "calabresa",
        "calareza": "calabresa",
        "calarezas": "calabresa",
        "mucarelas": "mucarela",
        "mussarelas": "mucarela",
        "muçarelas": "mucarela",
        "requijao": "requeijao",
        "requijoes": "requeijao",
        "requeijoes": "requeijao",
        "prezunto": "presunto",
        "prezuntos": "presunto",
        "gorgonzla": "gorgonzola",
        "gorgonzolas": "gorgonzola",
        "azeitonas": "azeitona",
        "bartata": "barato",
        "bartato": "barato",
        "masi": "mais",
    }
    if t in mapa_forte:
        return mapa_forte[t]

    vocab = termos_importantes_catalogo()
    if not vocab:
        return t

    match = process.extractOne(t, vocab, scorer=fuzz.ratio)
    if match and match[1] >= 86:
        return match[0]

    return t


def corrigir_frase_por_catalogo(texto: str) -> str:
    n = corrigir_busca(texto)
    tokens = []
    for token in n.split():
        tokens.append(corrigir_termo_por_catalogo(token))
    return " ".join(tokens).strip()


def tokens_busca_inteligente(texto: str) -> List[str]:
    """
    Tokens para busca real no catálogo.
    Preserva termos técnicos como: 9, mm, sem, com, amido, fatiada.
    Remove apenas palavras de comando.
    """
    n = corrigir_busca(texto or "")
    n = normalizar(n)

    ruido = {
        "pesquisa", "pesquisar", "busca", "buscar", "cotacao", "orcamento",
        "quero", "queria", "mande", "manda", "traga", "traz", "me",
        "por", "favor", "lista", "liste", "mostre", "procure", "procurar",
        "mais", "barato", "barata", "baratos", "baratas", "menor",
        "preco", "precos", "desconto", "total", "de", "da", "do", "das",
        "dos", "o", "a", "os", "as", "um", "uma", "uns", "umas", "para",
        "pra"
    }

    tokens = []
    for t in n.split():
        if not t or t in ruido:
            continue
        tokens.append(t)

    return tokens

def score_busca_produto(nome_normalizado: str, tokens: List[str], frase_normalizada: str = "") -> int:
    """
    Score para busca global.
    Prioriza produtos que contenham todos ou quase todos os tokens digitados.
    """
    if not nome_normalizado:
        return 0

    n = str(nome_normalizado)
    score = 0

    if frase_normalizada and frase_normalizada in n:
        score += 1000

    hits = 0

    for t in tokens:
        if not t:
            continue

        # Exato como palavra
        if re.search(rf"\b{re.escape(t)}\b", n):
            hits += 1
            score += 120
            continue

        # Substring
        if t in n:
            hits += 1
            score += 80
            continue

        # Casos técnicos: 9mm vs 9 mm
        if t.isdigit() and re.search(rf"\b{re.escape(t)}\s*mm\b", n):
            hits += 1
            score += 90
            continue

        try:
            sim = fuzz.partial_ratio(t, n)
            if sim >= 92:
                hits += 1
                score += 35
            elif sim >= 86:
                score += 12
        except Exception:
            pass

    if tokens:
        if hits == len(tokens):
            score += 500
        elif hits >= max(1, len(tokens) - 1):
            score += 250
        elif hits >= max(1, len(tokens) // 2):
            score += 80

    return score

def busca_fuzzy_universal(intent: IntencaoItem, limite: int = 80) -> pd.DataFrame:
    """
    Busca global na tabela inteira.

    Correção principal:
    - Para buscas específicas como "batata congelada 9 mm", exige os tokens
      principais dentro do produto antes de ordenar.
    - Não deixa a busca cair em apenas 1 item por fuzzy.
    - Não ordena por preço antes de garantir relevância.
    """
    if catalogo is None:
        return pd.DataFrame()

    frase_base = " ".join([
        str(intent.linha_original or ""),
        str(intent.produto or ""),
        " ".join(intent.marcas or []),
        str(intent.produto_base or ""),
    ])

    frase = normalizar(corrigir_frase_por_catalogo(frase_base))
    tokens = tokens_busca_inteligente(frase)

    # Remove quantidade inicial quando ela veio do pedido.
    # Ex: "5 batata congelada 9 mm" -> remove o 5, preserva 9 mm.
    if tokens and intent.quantidade:
        try:
            qtd_int = str(int(float(intent.quantidade)))
            if tokens[0] == qtd_int and str(intent.linha_original).strip().startswith(qtd_int):
                tokens = tokens[1:]
        except Exception:
            pass

    if intent.produto_base:
        base_norm = normalizar(intent.produto_base)
        if base_norm not in tokens:
            tokens.insert(0, base_norm)

    # Preserva obrigatórios reais: com/sem amido etc.
    for termo in intent.obrigatorios:
        tn = normalizar(termo)
        if tn and tn not in tokens:
            tokens.append(tn)

    if not tokens:
        return pd.DataFrame()

    df = catalogo.copy()

    def hit_count(nome: str) -> int:
        n = str(nome)
        total = 0
        for t in tokens:
            if re.search(rf"\b{re.escape(t)}\b", n) or t in n:
                total += 1
        return total

    df["_token_hits"] = df["NORM"].astype(str).apply(hit_count)
    df["_score"] = df["NORM"].astype(str).apply(
        lambda n: score_busca_produto(n, tokens, frase)
    )

    # Busca específica: exige mais precisão.
    # Ex: batata congelada 9 mm -> precisa bater todos ou quase todos.
    if len(tokens) >= 4:
        minimo = len(tokens) - 1
    elif len(tokens) == 3:
        minimo = 2
    else:
        minimo = 1

    df_filtrado = df[df["_token_hits"] >= minimo].copy()

    # Se ficou vazio, relaxa, mas ainda usa score.
    if df_filtrado.empty:
        df_filtrado = df[df["_score"] > 0].copy()

    if df_filtrado.empty:
        return pd.DataFrame()

    # Excluir termos explícitos
    for termo in intent.excluir:
        tn = normalizar(termo)
        if tn:
            df_filtrado = df_filtrado[~df_filtrado["NORM"].str.contains(tn, na=False)]

    # Obrigatórios explícitos
    for termo in intent.obrigatorios:
        tn = normalizar(termo)
        if tn:
            df_req = df_filtrado[df_filtrado["NORM"].str.contains(tn, na=False)]
            if not df_req.empty:
                df_filtrado = df_req

    df_filtrado = df_filtrado.sort_values(
        ["_token_hits", "_score"],
        ascending=[False, False]
    )

    return df_filtrado.head(limite).reset_index(drop=True)

def carregar_tabela_por_arquivo(arquivo: str) -> None:
    """
    Carrega CSV/XLSX sem abrir janela. Usado pela API Flask.
    """
    global tabela, catalogo

    if not arquivo:
        raise ValueError("Arquivo da tabela não informado.")

    df = carregar_csv_ou_excel(arquivo)
    tabela = padronizar_colunas(df)
    catalogo = enriquecer_catalogo(tabela)


def catalogo_carregado() -> bool:
    return catalogo is not None and len(catalogo) > 0


def total_produtos_catalogo() -> int:
    if catalogo is None:
        return 0
    return int(len(catalogo))


# =========================================================
# APRENDIZADO
# =========================================================
def carregar_aprendizado() -> Dict[str, str]:
    path = caminho_local(ARQUIVO_APRENDIZADO)
    if not os.path.exists(path):
        return {}
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return {}


def salvar_aprendizado(chave: str, produto: str) -> None:
    data = carregar_aprendizado()
    data[normalizar(chave)] = produto
    with open(caminho_local(ARQUIVO_APRENDIZADO), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def buscar_aprendizado(chave: str) -> Optional[pd.Series]:
    if catalogo is None:
        return None
    data = carregar_aprendizado()
    produto_nome = data.get(normalizar(chave))
    if not produto_nome:
        return None
    achou = catalogo[catalogo["PRODUTO"] == produto_nome]
    if not achou.empty:
        return achou.iloc[0]
    return None


# =========================================================
# INTERPRETAÇÃO DE PEDIDO
# =========================================================
@dataclass
class IntencaoItem:
    quantidade: float
    unidade: Optional[str]
    produto: str
    produto_base: Optional[str]
    marcas: List[str]
    excluir: List[str]
    obrigatorios: List[str]
    mais_barato: bool
    desconto: bool
    modo_lista: bool
    linha_original: str
    quantidade_resultados: int = 10


def dividir_itens_texto(texto: str) -> List[str]:
    bruto = []
    for linha in texto.splitlines():
        linha = linha.strip(" -*•\t")
        if not linha:
            continue

        linha = re.sub(r"^\s*(pesquisa|pesquisar|busca|buscar|cotacao|cotação|orcamento|orçamento)\s+", "", linha, flags=re.I)

        # divide por vírgula quando parece novo item
        partes = re.split(r",\s*(?=\d+\s|\bquero\b|\bme\b|\blista\b|\btotal\b|\bpesquisa\b)", linha, flags=re.I)
        bruto.extend([p.strip() for p in partes if p.strip()])

    if not bruto:
        bruto = [texto.strip()]
    return bruto


def extrair_quantidade_unidade(linha: str) -> Tuple[float, Optional[str], str]:
    """
    REGRA PMG:
    - Número no começo da linha = quantidade que o cliente quer comprar.
      Ex: "5 peças de muçarela scala" => quantidade 5 peças.
      Ex: "10 caixas muçarela scala" => quantidade 10 caixas.
    - Número depois do nome = característica do produto, NÃO quantidade.
      Ex: "azeitona média 7,5kg" => produto azeitona média 7,5kg, quantidade 1.
    """
    s = linha.strip()

    m = re.match(
        r"^\s*(\d+(?:[,.]\d+)?)\s*(caixas?|cx|kg|quilos?|kilo|bisnagas?|bis|bags?|bag|pacotes?|pct|fardos?|frd|fd|fdo|unidades?|un|pecas?|peças?|pç|pc|baldes?|bd|galoes|galões|gl|latas?|lt|vidros?|vd|litros?|l)?\s*(?:de\s+|do\s+|da\s+)?(.*)$",
        s,
        flags=re.I
    )

    if not m:
        return 1, None, s

    qtd = float(m.group(1).replace(",", "."))
    unidade = singular_unidade(m.group(2) or "") if m.group(2) else None
    resto = (m.group(3) or "").strip()

    if not resto:
        resto = s

    resto = re.sub(r"\s+", " ", resto)
    return qtd, unidade, resto


def interpretar_linha(linha: str) -> IntencaoItem:
    original_bruto = linha.strip()
    original = re.sub(r"^\s*(pesquisa|pesquisar|busca|buscar|cotacao|cotação|orcamento|orçamento)\s+", "", original_bruto, flags=re.I)
    original = normalizar_comando_inicial_quantidade(original)

    n = corrigir_frase_por_catalogo(original) if catalogo is not None else corrigir_busca(original)

    qtd, unidade, resto = extrair_quantidade_unidade(original)
    nresto = corrigir_frase_por_catalogo(resto) if catalogo is not None else corrigir_busca(resto)

    mais_barato = tem_modo_mais_barato(original)
    desconto = "desconto" in n
    modo_lista = ("lista" in n) or not mais_barato

    base = detectar_produto_base(nresto) or detectar_produto_base(n)

    excluir = []
    obrigatorios = []

    # exclusões explícitas
    if any(x in n for x in ["sem bufala", "sem bufula", "nao bufala", "nao quero bufala", "nao coloquei mulcarela de bufula"]):
        excluir.extend(["bufala", "bocconcino", "bolinha", "cereja", "cerejas"])

    # regra padrão muçarela: só traz búfala se pedir búfala
    if base == "mucarela" and "bufala" not in n and "bufulo" not in n and "bufula" not in n:
        excluir.extend(EXCLUSOES_PADRAO["mucarela"])

    if base == "presunto":
        excluir.extend(["apresuntado"])

    if "sem amido" in n:
        obrigatorios.extend(["sem", "amido"])
    if "com amido" in n:
        obrigatorios.extend(["com", "amido"])

    marcas = detectar_marcas(original)

    # Quando o usuário cita marcas como "tirolez, tres marias, scala", elas são filtro, não novo produto.
    # Se não achou base, tenta limpar marcas e detectar de novo.
    if not base:
        base = detectar_produto_base(" ".join([w for w in nresto.split() if w not in marcas]))

    # produto limpo para busca
    produto = nresto
    palavras_ruido = [
        "quero", "me", "mande", "manda", "traga", "traz", "lista", "total", "orcamento", "cotacao",
        "mais", "barata", "baratas", "barato", "baratos", "menor", "preco", "preço",
        "desconto", "com", "sem", "nao", "não", "quero", "tipos", "marcas",
    ]
    tokens = [t for t in produto.split() if t not in palavras_ruido and t not in PALAVRAS_MEDIDAS]
    produto_limpo = " ".join(tokens).strip()

    if base:
        produto_limpo = base

    return IntencaoItem(
        quantidade=qtd,
        unidade=unidade,
        produto=produto_limpo or nresto,
        produto_base=base,
        marcas=marcas,
        excluir=sorted(list(set(excluir))),
        obrigatorios=sorted(list(set(obrigatorios))),
        mais_barato=mais_barato,
        desconto=desconto,
        modo_lista=modo_lista,
        linha_original=original,
        quantidade_resultados=int(qtd) if mais_barato and qtd >= 1 else 10
    )


def interpretar_texto(texto: str) -> List[IntencaoItem]:
    # OpenAI é opcional. O motor principal é local para não inventar produto.
    linhas = dividir_itens_texto(texto)
    return [interpretar_linha(l) for l in linhas if l.strip()]


# =========================================================
# FILTRO E BUSCA
# =========================================================
def produto_bate_base(row: pd.Series, intent: IntencaoItem) -> bool:
    nome = str(row.get("NORM", ""))

    if intent.produto_base:
        base = intent.produto_base

        if base == "mucarela":
            return bool(re.search(r"\bmucarela\b", nome))
        if base == "presunto":
            return bool(re.search(r"\bpresunto\b", nome)) and "apresuntado" not in nome
        if base == "apresuntado":
            return "apresuntado" in nome
        if base == "requeijao":
            return "requeijao" in nome
        if base == "calabresa":
            return "calabresa" in nome
        if base == "azeitona":
            return "azeitona" in nome
        if base == "batata":
            return "batata" in nome
        if base == "molho tomate":
            return "molho" in nome and "tomate" in nome
        if base == "tomate pelado":
            return "tomate" in nome and "pelado" in nome

        termos = SINONIMOS.get(base, [base])
        return any(normalizar(t) in nome for t in termos)

    # Sem base conhecida: usa tokens em vez de fuzzy duro.
    # Isso permite achar "batata congelada 9 mm", "batata palito 9x18", etc.
    termo = corrigir_frase_por_catalogo(intent.produto or intent.linha_original)
    tokens = tokens_busca_inteligente(termo)

    if not tokens:
        return False

    score = score_busca_produto(nome, tokens, normalizar(termo))

    # score baixo já é suficiente aqui porque filtrar_catalogo depois ordena/refina.
    return score > 0




def tokens_especificos_intent(intent: IntencaoItem) -> List[str]:
    """
    Tokens que refinam uma busca com base conhecida.
    Ex:
    "30 batata palito mais barata" -> ["palito"]
    "batata congelada 9 mm" -> ["congelada", "9", "mm"]
    "mucarela scala" -> ["scala"]
    """
    texto = str(intent.linha_original or "")
    tokens = tokens_busca_inteligente(texto)

    # remove quantidade inicial, mas preserva medidas reais como 9 mm
    try:
        qtd = str(int(float(intent.quantidade)))
        if tokens and tokens[0] == qtd:
            tokens = tokens[1:]
    except Exception:
        pass

    remover = set()
    if intent.produto_base:
        remover.add(normalizar(intent.produto_base))
        for termo in SINONIMOS.get(intent.produto_base, []):
            for t in normalizar(termo).split():
                remover.add(t)

    # remove marcas já tratadas em outro filtro
    for marca in intent.marcas or []:
        for t in normalizar(marca).split():
            remover.add(t)

    # remove ruídos extras que nunca devem filtrar produto
    remover.update({
        "mais", "barato", "barata", "baratos", "baratas",
        "menor", "preco", "precos", "desconto", "lista"
    })

    finais = []
    for t in tokens:
        if t and t not in remover and t not in finais:
            finais.append(t)

    return finais


def filtrar_catalogo(intent: IntencaoItem) -> pd.DataFrame:
    if catalogo is None:
        return pd.DataFrame()

    # Para produtos sem base conhecida, usa busca global direto.
    if not intent.produto_base:
        df = busca_fuzzy_universal(intent, limite=120)
    else:
        df = catalogo.copy()
        df = df[df.apply(lambda r: produto_bate_base(r, intent), axis=1)].copy()

        # Refinamento por termos específicos.
        # Isso corrige "batata palito mais barata" para não trazer qualquer batata.
        termos_refino = tokens_especificos_intent(intent)

        if termos_refino and not df.empty:
            def hits_refino(nome: str) -> int:
                n = str(nome)
                total = 0
                for t in termos_refino:
                    if re.search(rf"\b{re.escape(t)}\b", n) or t in n:
                        total += 1
                return total

            df["_refino_hits"] = df["NORM"].astype(str).apply(hits_refino)

            # Para 1 termo, exige 1. Para vários, exige quase todos.
            minimo = 1 if len(termos_refino) <= 2 else len(termos_refino) - 1
            df_ref = df[df["_refino_hits"] >= minimo].copy()

            if not df_ref.empty:
                df = df_ref

    if df.empty:
        return busca_fuzzy_universal(intent, limite=120)

    # Obrigatórios: sem amido, com amido etc.
    for termo in intent.obrigatorios:
        tn = normalizar(termo)
        if tn:
            df_req = df[df["NORM"].str.contains(tn, na=False)]
            if not df_req.empty:
                df = df_req

    # Excluir termos
    for termo in intent.excluir:
        tn = normalizar(termo)
        if tn:
            df = df[~df["NORM"].str.contains(tn, na=False)]

    # Marcas: se tiver marcas citadas, aceita qualquer uma delas.
    if intent.marcas:
        marcas_norm = [normalizar(m) for m in intent.marcas]

        def marca_ok(nome):
            n = normalizar(nome)
            return any(m in n for m in marcas_norm)

        df_marca = df[df["PRODUTO"].apply(marca_ok)]

        if not df_marca.empty:
            df = df_marca

    if intent.unidade == "kg":
        df_kg = df[df["PRECO_KG"].notna()]
        if not df_kg.empty:
            df = df_kg

    return df.reset_index(drop=True)



def chave_ordenacao(row: pd.Series, intent: IntencaoItem) -> float:
    if intent.unidade == "kg" or intent.produto_base in ["mucarela", "presunto", "calabresa", "bacon", "manteiga"]:
        if not pd.isna(row.get("PRECO_KG")):
            return float(row["PRECO_KG"])
    if not pd.isna(row.get("PRECO_UNIDADE")):
        return float(row["PRECO_UNIDADE"])
    if not pd.isna(row.get("PRECO_CAIXA")):
        return float(row["PRECO_CAIXA"])
    return float(row["PRECO"])


def aplicar_desconto_valor(valor: Optional[float], desconto: bool) -> Optional[float]:
    if valor is None or pd.isna(valor):
        return None
    if desconto:
        return float(valor) * (1 - DESCONTO_PADRAO / 100)
    return float(valor)


def valores_row(row: pd.Series, desconto: bool) -> Dict[str, Optional[float]]:
    return {
        "preco_tabela": aplicar_desconto_valor(row["PRECO"], desconto),
        "preco_kg": aplicar_desconto_valor(row.get("PRECO_KG"), desconto),
        "preco_litro": aplicar_desconto_valor(row.get("PRECO_LITRO"), desconto),
        "preco_unidade": aplicar_desconto_valor(row.get("PRECO_UNIDADE"), desconto),
        "preco_caixa": aplicar_desconto_valor(row.get("PRECO_CAIXA"), desconto),
    }


def calcular_subtotal(row: pd.Series, intent: IntencaoItem) -> Tuple[float, str, float]:
    vals = valores_row(row, intent.desconto)
    qtd = intent.quantidade
    unidade = intent.unidade

    # Se usuário não informou unidade:
    # - Mais barato é lista/opção, não subtotal real.
    # - Cotação normal usa unidade de venda.
    if not unidade:
        unidade = row["VEND"]

    unidade = singular_unidade(unidade)

    if unidade == "caixa":
        preco = vals["preco_caixa"] or vals["preco_tabela"] or 0
        return qtd * preco, "caixa", preco

    if unidade == "fardo":
        preco = vals["preco_caixa"] or vals["preco_tabela"] or 0
        return qtd * preco, "fardo", preco

    if unidade == "kg":
        preco = vals["preco_kg"] or vals["preco_tabela"] or 0
        return qtd * preco, "kg", preco

    if unidade == "l":
        preco = vals["preco_litro"] or vals["preco_tabela"] or 0
        return qtd * preco, "litro", preco

    # bisnaga, bag, pacote, peça, unidade...
    preco = vals["preco_unidade"] or vals["preco_tabela"] or 0
    return qtd * preco, unidade, preco


def abrir_janela_escolha_produto(intent: IntencaoItem, df: pd.DataFrame) -> Optional[Any]:
    """
    Janela com rolagem para escolher até 20 produtos.
    Também permite pesquisar novamente.
    Retorna:
    - int: índice escolhido dentro do df
    - ("nova_busca", texto): quando usuário quer pesquisar outro termo
    - None: cancelou
    """
    escolha = {"valor": None}

    win = ctk.CTkToplevel(app)
    win.title("Escolha o produto")
    win.geometry("860x680")
    win.grab_set()

    ctk.CTkLabel(
        win,
        text=f"Pedido: {intent.linha_original}",
        font=("Arial", 15, "bold")
    ).pack(anchor="w", padx=14, pady=(12, 4))

    ctk.CTkLabel(
        win,
        text="Role a lista e clique no produto correto. Se não achou, pesquise novamente abaixo.",
        text_color="#cbd5e1"
    ).pack(anchor="w", padx=14, pady=(0, 8))

    busca_frame = ctk.CTkFrame(win)
    busca_frame.pack(fill="x", padx=14, pady=(0, 8))

    busca_entry = ctk.CTkEntry(busca_frame, placeholder_text="Pesquisar outro produto. Ex: atum tours 400g, azeitona média 7,5kg, muçarela tirolez...")
    busca_entry.pack(side="left", fill="x", expand=True, padx=(8, 6), pady=8)

    def pesquisar_novamente():
        termo = busca_entry.get().strip()
        if termo:
            escolha["valor"] = ("nova_busca", termo)
            win.destroy()

    ctk.CTkButton(
        busca_frame,
        text="Pesquisar novamente",
        command=pesquisar_novamente,
        height=34,
        fg_color="#16a34a",
        width=170
    ).pack(side="right", padx=(6, 8), pady=8)

    scroll = ctk.CTkScrollableFrame(win, width=820, height=480)
    scroll.pack(fill="both", expand=True, padx=14, pady=8)

    def selecionar(i: int):
        escolha["valor"] = i
        win.destroy()

    for i, row in df.iterrows():
        vals = valores_row(row, intent.desconto)

        card = ctk.CTkFrame(scroll)
        card.pack(fill="x", padx=6, pady=6)

        titulo = f"{i+1}. {row['PRODUTO']}"
        ctk.CTkLabel(
            card,
            text=titulo,
            font=("Arial", 14, "bold"),
            anchor="w",
            justify="left"
        ).pack(fill="x", padx=10, pady=(8, 2))

        linhas = [f"Código: {row['COD']} | Vendido por: {row['VEND_POR']}"]

        if vals["preco_kg"] is not None:
            linhas.append(f"Valor do kg: {dinheiro(vals['preco_kg'])}")

        if vals["preco_unidade"] is not None:
            un = nome_unidade_cliente(row.get("EMB_UN_INTERNA") or row["VEND"])
            linhas.append(f"Valor do(a) {un}: {dinheiro(vals['preco_unidade'])}")

        if vals["preco_caixa"] is not None:
            cont = nome_unidade_cliente(row.get("EMB_CONTAINER") or "caixa")
            linhas.append(f"Valor da/do {cont}: {dinheiro(vals['preco_caixa'])}")

        ctk.CTkLabel(
            card,
            text="\n".join(linhas),
            anchor="w",
            justify="left",
            text_color="#e5e7eb"
        ).pack(fill="x", padx=10, pady=4)

        ctk.CTkButton(
            card,
            text=f"Escolher opção {i+1}",
            command=lambda idx=i: selecionar(idx),
            height=32,
            fg_color="#2563eb"
        ).pack(anchor="e", padx=10, pady=(2, 8))

    botoes = ctk.CTkFrame(win)
    botoes.pack(fill="x", padx=14, pady=(0, 12))

    ctk.CTkButton(
        botoes,
        text="Cancelar",
        command=win.destroy,
        fg_color="#64748b",
        height=36
    ).pack(side="right", padx=6)

    win.wait_window()
    return escolha["valor"]


def escolher_produto_lista(intent: IntencaoItem) -> Optional[pd.Series]:
    """
    Versão API/web.
    Não abre janela Tkinter.
    Escolhe automaticamente a melhor opção ordenada pelo menor preço.
    """
    apr = buscar_aprendizado(intent.linha_original)
    if apr is not None:
        return apr

    df = filtrar_catalogo(intent)

    if df.empty:
        df = busca_fuzzy_universal(intent, limite=MAX_OPCOES_LISTA)

    if df.empty:
        return None

    df = df.copy()
    df["_ordem"] = df.apply(lambda r: chave_ordenacao(r, intent), axis=1)
    df = df.sort_values("_ordem").head(MAX_OPCOES_LISTA).reset_index(drop=True)

    row = df.iloc[0]
    salvar_aprendizado(intent.linha_original, row["PRODUTO"])

    return row



def emoji_produto(nome: str) -> str:
    n = normalizar(nome)
    if any(x in n for x in ["mucarela", "queijo", "parmesao", "provolone", "gorgonzola", "prato"]):
        return "🧀"
    if any(x in n for x in ["requeijao", "catupiry", "cream cheese", "cheddar", "leite", "manteiga"]):
        return "🥛"
    if any(x in n for x in ["calabresa", "presunto", "bacon", "lombo", "pepperoni", "salame", "carne"]):
        return "🥩"
    if any(x in n for x in ["azeitona", "tomate", "alho", "brocolis", "palmito", "milho"]):
        return "🥫"
    if any(x in n for x in ["farinha", "arroz", "feijao", "acucar", "sal"]):
        return "🌾"
    if any(x in n for x in ["coca", "fanta", "guarana", "agua", "suco"]):
        return "🥤"
    if any(x in n for x in ["oleo", "azeite"]):
        return "🛢️"
    if any(x in n for x in ["atum", "sardinha", "peixe", "salmao"]):
        return "🐟"
    return "📦"


def cabecalho_cotacao(cliente: str) -> List[str]:
    return [
        "📋 *COTAÇÃO PMG*",
        f"👤 Cliente: {cliente}",
        "",
        "Segue cotação conforme solicitado:",
        ""
    ]


def rodape_cotacao(total: float) -> List[str]:
    return [
        "━━━━━━━━━━━━━━━━━━━━",
        f"💰 *TOTAL: {dinheiro(total)}*",
        "",
        "✅ Valores sujeitos à disponibilidade de estoque.",
        "📦 Valores de caixa/fardo informados para facilitar a conferência."
    ]

# =========================================================
# RESPOSTA E COTAÇÃO
# =========================================================
def linhas_preco_cliente(row: pd.Series, intent: IntencaoItem) -> List[str]:
    vals = valores_row(row, intent.desconto)
    linhas = []

    if vals["preco_kg"] is not None:
        linhas.append(f"   Valor do kg: {dinheiro(vals['preco_kg'])}")

    if vals["preco_litro"] is not None:
        linhas.append(f"   Valor do litro: {dinheiro(vals['preco_litro'])}")

    unidade_interna = row.get("EMB_UN_INTERNA")
    if vals["preco_unidade"] is not None:
        nome = nome_unidade_cliente(unidade_interna or row["VEND"])
        if nome == "peça":
            linhas.append(f"   Valor da peça: {dinheiro(vals['preco_unidade'])}")
        elif nome == "bisnaga":
            linhas.append(f"   Valor da bisnaga: {dinheiro(vals['preco_unidade'])}")
        elif nome == "bag":
            linhas.append(f"   Valor da bag: {dinheiro(vals['preco_unidade'])}")
        elif nome == "pacote":
            linhas.append(f"   Valor do pacote: {dinheiro(vals['preco_unidade'])}")
        elif nome == "unidade":
            linhas.append(f"   Valor da unidade: {dinheiro(vals['preco_unidade'])}")
        else:
            linhas.append(f"   Valor do(a) {nome}: {dinheiro(vals['preco_unidade'])}")

    if vals["preco_caixa"] is not None:
        cont = nome_unidade_cliente(row.get("EMB_CONTAINER") or "caixa")
        if cont == "fardo":
            linhas.append(f"   Valor do fardo: {dinheiro(vals['preco_caixa'])}")
        else:
            linhas.append(f"   Valor da caixa: {dinheiro(vals['preco_caixa'])}")

    return linhas


def gerar_opcoes_mais_baratas(intent: IntencaoItem) -> str:
    df = filtrar_catalogo(intent)
    if df.empty:
        df = busca_fuzzy_universal(intent)
    if df.empty:
        return f"Não encontrei opções para: {intent.linha_original}"

    df = df.copy()
    df["_ordem"] = df.apply(lambda r: chave_ordenacao(r, intent), axis=1)
    df = df.sort_values("_ordem").head(intent.quantidade_resultados).reset_index(drop=True)

    titulo = intent.produto_base or intent.produto
    linhas = [
        f"🔎 *OPÇÕES MAIS BARATAS — {titulo.upper()}*",
        "",
    ]

    for i, row in df.iterrows():
        vals = valores_row(row, intent.desconto)
        emoji = emoji_produto(row["PRODUTO"])

        linhas.append(f"{emoji} *{i+1}. {row['PRODUTO']}*")
        if vals["preco_kg"] is not None:
            linhas.append(f"   ⚖️ Kg: {dinheiro(vals['preco_kg'])}")
        if vals["preco_unidade"] is not None:
            un = nome_unidade_cliente(row.get("EMB_UN_INTERNA") or row["VEND"])
            linhas.append(f"   🔹 {un.capitalize()}: {dinheiro(vals['preco_unidade'])}")
        if vals["preco_caixa"] is not None:
            cont = nome_unidade_cliente(row.get("EMB_CONTAINER") or "caixa")
            label = "Caixa" if cont == "caixa" else "Fardo" if cont == "fardo" else cont.capitalize()
            linhas.append(f"   📦 {label}: {dinheiro(vals['preco_caixa'])}")
        linhas.append("")

    linhas.append("✅ Valores ordenados do menor para o maior.")
    return "\n".join(linhas).strip()


def gerar_cotacao_automatica(cliente: str, itens: List[Tuple[IntencaoItem, pd.Series]]) -> Tuple[str, str]:
    linhas_cliente = cabecalho_cotacao(cliente)
    linhas_interno = [f"COTAÇÃO INTERNA - {cliente}", ""]
    total = 0.0

    for idx, (intent, row) in enumerate(itens, start=1):
        subtotal, unidade_final, preco_aplicado = calcular_subtotal(row, intent)
        total += subtotal

        qtd_txt = fmt_num(intent.quantidade)
        unidade_txt = plural_unidade(unidade_final) if intent.quantidade != 1 and unidade_final != "kg" else nome_unidade_cliente(unidade_final)
        emoji = emoji_produto(row["PRODUTO"])

        linhas_cliente.append(f"{emoji} *ITEM {idx}*")
        linhas_cliente.append(f"{qtd_txt} {unidade_txt} - {row['PRODUTO']}")

        for linha in linhas_preco_cliente(row, intent):
            linha = linha.strip()
            linha = linha.replace("Valor do kg:", "⚖️ Kg:")
            linha = linha.replace("Valor do litro:", "🥤 Litro:")
            linha = linha.replace("Valor da peça:", "🧩 Peça:")
            linha = linha.replace("Valor da bisnaga:", "🥛 Bisnaga:")
            linha = linha.replace("Valor da bag:", "🧴 Bag:")
            linha = linha.replace("Valor do pacote:", "📦 Pacote:")
            linha = linha.replace("Valor da unidade:", "🔹 Unidade:")
            linha = linha.replace("Valor da caixa:", "📦 Caixa:")
            linha = linha.replace("Valor do fardo:", "📦 Fardo:")
            linha = linha.replace("Valor da/do caixa:", "📦 Caixa:")
            linha = linha.replace("Valor da/do fardo:", "📦 Fardo:")
            linhas_cliente.append(f"   {linha}")

        linhas_cliente.append(f"   💵 Subtotal: *{dinheiro(subtotal)}*")
        linhas_cliente.append("")

        linhas_interno.append(f"{idx}. {intent.linha_original}")
        linhas_interno.append(f"   Produto escolhido: {row['PRODUTO']} | COD {row['COD']}")
        linhas_interno.append(f"   Unidade aplicada: {unidade_final} | Preço aplicado: {dinheiro(preco_aplicado)}")
        linhas_interno.append(f"   Subtotal: {dinheiro(subtotal)}")
        linhas_interno.append("")

    linhas_cliente.extend(rodape_cotacao(total))
    linhas_interno.append(f"TOTAL: {dinheiro(total)}")

    return "\n".join(linhas_cliente), "\n".join(linhas_interno)


def processar_pedido(cliente: str, texto: str) -> Tuple[str, str]:
    intents = interpretar_texto(texto)

    # Se tiver pedido "mais barato", ele deve retornar opções, não fechar cotação automática.
    # Isso evita transformar "5 muçarelas mais baratas" em 5 kg de uma muçarela só.
    intents_mais_barato = [i for i in intents if i.mais_barato]
    intents_cotacao = [i for i in intents if not i.mais_barato]

    blocos_mais_barato = []
    if intents_mais_barato:
        blocos_mais_barato = [gerar_opcoes_mais_baratas(i) for i in intents_mais_barato]

    if intents_mais_barato and not intents_cotacao:
        resp = ("\n\n" + "-" * 50 + "\n\n").join(blocos_mais_barato)
        return resp.strip(), resp.strip()

    itens_escolhidos = []
    erros = []
    intents = intents_cotacao

    for intent in intents:
        if intent.mais_barato:
            # Se misturou cotação com mais barato, traz primeira opção mais barata como item.
            df = filtrar_catalogo(intent)
            if df.empty:
                erros.append(f"Não encontrei: {intent.linha_original}")
                continue
            df = df.copy()
            df["_ordem"] = df.apply(lambda r: chave_ordenacao(r, intent), axis=1)
            row = df.sort_values("_ordem").iloc[0]
            itens_escolhidos.append((intent, row))
        else:
            row = escolher_produto_lista(intent)
            if row is None:
                erros.append(f"Não encontrei ou não foi escolhido: {intent.linha_original}")
            else:
                itens_escolhidos.append((intent, row))

    if not itens_escolhidos:
        return "\n".join(erros) if erros else "Nenhum item encontrado.", "\n".join(erros)

    cliente_txt, interno_txt = gerar_cotacao_automatica(cliente, itens_escolhidos)

    if erros:
        cliente_txt += "\n\nItens não encontrados:\n" + "\n".join(f"- {e}" for e in erros)
        interno_txt += "\n\nErros:\n" + "\n".join(f"- {e}" for e in erros)

    if blocos_mais_barato:
        opcoes_txt = ("\n\n" + "-" * 50 + "\n\n").join(blocos_mais_barato)
        cliente_txt = opcoes_txt + "\n\n" + "=" * 50 + "\n\n" + cliente_txt
        interno_txt = opcoes_txt + "\n\n" + "=" * 50 + "\n\n" + interno_txt

    return cliente_txt, interno_txt


# =========================================================
# CLIENTES
# =========================================================
def carregar_clientes() -> List[str]:
    path = caminho_local(ARQUIVO_CLIENTES)
    if not os.path.exists(path):
        return []
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return []


def salvar_cliente(nome: str) -> None:
    nome = nome.strip()
    if not nome:
        return
    clientes = sorted(list(set(carregar_clientes() + [nome])))
    with open(caminho_local(ARQUIVO_CLIENTES), "w", encoding="utf-8") as f:
        json.dump(clientes, f, ensure_ascii=False, indent=2)


# =========================================================
# FLUXO WEB COM ESCOLHA MANUAL DE PRODUTOS
# =========================================================
def _row_para_opcao_web(row: pd.Series, intent: IntencaoItem, index: int) -> Dict[str, Any]:
    vals = valores_row(row, intent.desconto)

    precos = []
    if vals.get("preco_kg") is not None:
        precos.append({"label": "Kg", "valor": dinheiro(vals["preco_kg"])})
    if vals.get("preco_litro") is not None:
        precos.append({"label": "Litro", "valor": dinheiro(vals["preco_litro"])})
    if vals.get("preco_unidade") is not None:
        un = nome_unidade_cliente(row.get("EMB_UN_INTERNA") or row["VEND"])
        precos.append({"label": un.capitalize(), "valor": dinheiro(vals["preco_unidade"])})
    if vals.get("preco_caixa") is not None:
        cont = nome_unidade_cliente(row.get("EMB_CONTAINER") or "caixa")
        label = "Caixa" if cont == "caixa" else "Fardo" if cont == "fardo" else cont.capitalize()
        precos.append({"label": label, "valor": dinheiro(vals["preco_caixa"])})

    return {
        "index": index,
        "produto": str(row["PRODUTO"]),
        "cod": str(row.get("COD", "")),
        "vend_por": str(row.get("VEND_POR", "")),
        "precos": precos,
    }


def buscar_produtos_web(termo: str, limite: int = 60) -> List[Dict[str, Any]]:
    """
    Busca GLOBAL na tabela carregada.
    Essa função é usada pelo botão "Pesquisar geral" do modal.
    Ela SEMPRE pesquisa na tabela inteira, não nas opções abertas.
    """
    if catalogo is None:
        raise ValueError("Carregue a tabela PDF/CSV/XLSX antes de pesquisar.")

    termo = (termo or "").strip()

    if not termo:
        return []

    intent = interpretar_linha(termo)

    df = busca_fuzzy_universal(intent, limite=limite)

    if df.empty:
        df = filtrar_catalogo(intent)

    if df.empty:
        return []

    df = df.copy()

    if "_token_hits" in df.columns and "_score" in df.columns:
        df = df.sort_values(["_token_hits", "_score"], ascending=[False, False])
    elif "_score" in df.columns:
        df = df.sort_values("_score", ascending=False)
    else:
        df["_ordem"] = df.apply(lambda r: chave_ordenacao(r, intent), axis=1)
        df = df.sort_values("_ordem")

    df = df.head(limite).reset_index(drop=True)

    return [
        _row_para_opcao_web(row, intent, i)
        for i, row in df.iterrows()
    ]

def preparar_cotacao_web(texto: str) -> Dict[str, Any]:
    """
    Prepara a cotação para o CRM.

    Regra importante:
    - Se o usuário pedir "mais barato" / "mais baratas", NÃO abre modal.
      Retorna a lista de opções igual ao desktop:
      "5 mussarelas mais baratas" => 5 tipos/opções de muçarela, não 5 kg.
    - Se for cotação normal, abre modal para escolha manual do produto.
    """
    if catalogo is None:
        raise ValueError("Carregue a tabela CSV/XLSX antes de gerar a cotação.")

    intents = interpretar_texto(texto)
    itens = []
    blocos_mais_barato = []

    for item_index, intent in enumerate(intents):
        if intent.mais_barato:
            blocos_mais_barato.append(gerar_opcoes_mais_baratas(intent))
            continue

        # Para cotação normal, abre modal com opções mais relevantes primeiro.
        # Não ordena por preço antes de relevância; isso estava escondendo itens específicos.
        df = busca_fuzzy_universal(intent, limite=60)

        if df.empty:
            df = filtrar_catalogo(intent)

        if not df.empty:
            df = df.copy()

            if "_token_hits" in df.columns and "_score" in df.columns:
                df = df.sort_values(["_token_hits", "_score"], ascending=[False, False])
            elif "_score" in df.columns:
                df = df.sort_values("_score", ascending=False)
            else:
                df["_ordem"] = df.apply(lambda r: chave_ordenacao(r, intent), axis=1)
                df = df.sort_values("_ordem")

            df = df.head(MAX_OPCOES_LISTA).reset_index(drop=True)

        opcoes = []
        for i, row in df.iterrows():
            opcoes.append(_row_para_opcao_web(row, intent, i))

        itens.append({
            "item_index": item_index,
            "linha_original": intent.linha_original,
            "quantidade": intent.quantidade,
            "unidade": intent.unidade,
            "produto": intent.produto,
            "mais_barato": intent.mais_barato,
            "desconto": intent.desconto,
            "opcoes": opcoes,
        })

    resultado_mais_barato = ("\n\n" + "-" * 50 + "\n\n").join(blocos_mais_barato).strip()

    # Se só pediu lista de mais baratos, já devolve resultado direto.
    if resultado_mais_barato and not itens:
        return {
            "itens": [],
            "resultado_direto": resultado_mais_barato,
            "interno_direto": resultado_mais_barato,
        }

    # Se misturou lista + cotação normal, guarda para juntar no final.
    return {
        "itens": itens,
        "resultado_inicial": resultado_mais_barato,
        "interno_inicial": resultado_mais_barato,
    }


def finalizar_cotacao_web(cliente: str, texto: str, escolhas: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Finaliza a cotação usando os produtos escolhidos no CRM.

    Regras:
    - Itens normais precisam de escolha manual.
    - Itens "mais baratos" não precisam de escolha; geram lista de opções.
    """
    if catalogo is None:
        raise ValueError("Carregue a tabela CSV/XLSX antes de gerar a cotação.")

    intents = interpretar_texto(texto)

    escolhas_por_item = {
        int(e.get("item_index")): str(e.get("produto"))
        for e in escolhas
        if e.get("item_index") is not None and e.get("produto")
    }

    itens_escolhidos = []
    erros = []
    blocos_mais_barato = []

    for item_index, intent in enumerate(intents):
        if intent.mais_barato:
            blocos_mais_barato.append(gerar_opcoes_mais_baratas(intent))
            continue

        produto_escolhido = escolhas_por_item.get(item_index)

        if not produto_escolhido:
            erros.append(f"Produto não escolhido: {intent.linha_original}")
            continue

        achou = catalogo[catalogo["PRODUTO"] == produto_escolhido]

        if achou.empty:
            erros.append(f"Produto escolhido não encontrado na tabela: {produto_escolhido}")
            continue

        row = achou.iloc[0]
        salvar_aprendizado(intent.linha_original, row["PRODUTO"])
        itens_escolhidos.append((intent, row))

    bloco_mais_barato_txt = ("\n\n" + "-" * 50 + "\n\n").join(blocos_mais_barato).strip()

    if not itens_escolhidos:
        if bloco_mais_barato_txt:
            return bloco_mais_barato_txt, bloco_mais_barato_txt

        erro_txt = "\n".join(erros) if erros else "Nenhum item escolhido."
        return erro_txt, erro_txt

    cliente_txt, interno_txt = gerar_cotacao_automatica(cliente, itens_escolhidos)

    if bloco_mais_barato_txt:
        cliente_txt = bloco_mais_barato_txt + "\n\n" + "=" * 50 + "\n\n" + cliente_txt
        interno_txt = bloco_mais_barato_txt + "\n\n" + "=" * 50 + "\n\n" + interno_txt

    if erros:
        cliente_txt += "\n\nItens com problema:\n" + "\n".join(f"- {e}" for e in erros)
        interno_txt += "\n\nItens com problema:\n" + "\n".join(f"- {e}" for e in erros)

    return cliente_txt, interno_txt

