import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from cotador import (
    buscar_produtos_web,
    carregar_clientes,
    carregar_tabela_por_arquivo,
    finalizar_cotacao_web,
    preparar_cotacao_web,
    salvar_cliente,
)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "message": "API Cotador PMG rodando no Railway"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "service": "cotador-pmg",
        "status": "online"
    })


@app.route("/upload-tabela", methods=["POST"])
def upload_tabela():
    try:
        arquivo = request.files.get("file")

        if not arquivo:
            return jsonify({
                "ok": False,
                "erro": "Nenhum arquivo enviado"
            }), 400

        caminho = os.path.join(UPLOAD_DIR, arquivo.filename)
        arquivo.save(caminho)

        carregar_tabela_por_arquivo(caminho)

        return jsonify({
            "ok": True,
            "mensagem": "Tabela carregada com sucesso"
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "erro": str(e)
        }), 500


@app.route("/clientes", methods=["GET"])
def listar_clientes():
    try:
        return jsonify({
            "ok": True,
            "clientes": carregar_clientes()
        })
    except Exception as e:
        return jsonify({
            "ok": False,
            "clientes": [],
            "erro": str(e)
        }), 500


@app.route("/clientes", methods=["POST"])
def salvar_cliente_api():
    try:
        data = request.get_json() or {}
        nome = data.get("nome", "").strip()

        if not nome:
            return jsonify({
                "ok": False,
                "erro": "Nome do cliente vazio"
            }), 400

        salvar_cliente(nome)

        return jsonify({
            "ok": True,
            "mensagem": "Cliente salvo com sucesso"
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "erro": str(e)
        }), 500


@app.route("/preparar-cotacao", methods=["POST"])
def preparar_cotacao():
    try:
        data = request.get_json() or {}
        pedido = data.get("pedido", "").strip()

        if not pedido:
            return jsonify({
                "ok": False,
                "erro": "Digite ou cole um pedido antes de gerar a cotação."
            }), 400

        preparo = preparar_cotacao_web(pedido)

        return jsonify({
            "ok": True,
            **preparo
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "erro": str(e)
        }), 500


@app.route("/buscar-produtos", methods=["POST"])
def buscar_produtos():
    try:
        data = request.get_json() or {}
        termo = data.get("termo", "").strip()

        if not termo:
            return jsonify({
                "ok": True,
                "opcoes": []
            })

        opcoes = buscar_produtos_web(termo)

        return jsonify({
            "ok": True,
            "opcoes": opcoes
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "opcoes": [],
            "erro": str(e)
        }), 500


@app.route("/finalizar-cotacao", methods=["POST"])
def finalizar_cotacao():
    try:
        data = request.get_json() or {}

        cliente = data.get("cliente", "Cliente").strip() or "Cliente"
        pedido = data.get("pedido", "").strip()
        escolhas = data.get("escolhas", [])

        if not pedido:
            return jsonify({
                "ok": False,
                "resultado": "Digite ou cole um pedido antes de gerar a cotação.",
                "interno": ""
            }), 400

        salvar_cliente(cliente)

        resultado, interno = finalizar_cotacao_web(cliente, pedido, escolhas)

        return jsonify({
            "ok": True,
            "resultado": resultado,
            "interno": interno
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "resultado": f"Erro ao processar cotação: {str(e)}",
            "interno": f"Erro interno: {str(e)}"
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )