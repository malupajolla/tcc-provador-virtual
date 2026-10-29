"""Analisa os CSVs gerados por `python main.py --medir`.

Uso:
    python analisar.py desempenho medicoes\\arquivo1.csv medicoes\\arquivo2.csv ...
    python analisar.py jitter medicoes\\arquivo.csv
"""
import csv
import sys

import numpy as np

DESCARTE_INICIO_S = 2.0   # ignora o inicio (aquecimento do modelo e calibracao do pitch)
DESCARTE_TROCA_S = 1.0    # ignora 1 s depois de ligar/desligar o filtro (filtro reiniciado)
LIMITE_MS = 40.0


def ler(caminho):
    with open(caminho, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f, delimiter=";"))
    t0 = float(linhas[0]["t_s"])
    return [l for l in linhas if float(l["t_s"]) - t0 >= DESCARTE_INICIO_S]


def coluna(linhas, nome):
    return np.array([float(l[nome]) for l in linhas])


def desempenho(caminhos):
    todos_proc = []
    for caminho in caminhos:
        linhas = ler(caminho)
        t = coluna(linhas, "t_s")
        proc = coluna(linhas, "proc_ms")
        fps = (len(linhas) - 1) / (t[-1] - t[0])
        todos_proc.append(proc)
        print(f"{caminho}: {len(linhas)} quadros em {t[-1] - t[0]:.1f} s -> {fps:.1f} FPS")

    proc = np.concatenate(todos_proc)
    print(f"\nTotal: {len(proc)} quadros em {len(caminhos)} execucoes")
    print(f"Tempo de processamento (ms): media {proc.mean():.1f} | mediana {np.median(proc):.1f} "
          f"| p95 {np.percentile(proc, 95):.1f} | maximo {proc.max():.1f}")
    print(f"Quadros acima de {LIMITE_MS:.0f} ms: {(proc > LIMITE_MS).mean() * 100:.1f}%")


def jitter(caminho):
    linhas = [l for l in ler(caminho) if l["rosto_detectado"] == "1" and l["ancora_x"] != ""]

    for valor, nome in (("1", "COM filtro"), ("0", "SEM filtro")):
        grupo = [l for l in linhas if l["filtro"] == valor]
        if not grupo:
            print(f"{nome}: nenhum quadro")
            continue
        t0 = float(grupo[0]["t_s"])
        grupo = [l for l in grupo if float(l["t_s"]) - t0 >= DESCARTE_TROCA_S]

        x, y = coluna(grupo, "ancora_x"), coluna(grupo, "ancora_y")
        desvio = np.sqrt(x.std() ** 2 + y.std() ** 2)
        print(f"{nome}: {len(grupo)} quadros | desvio x {x.std():.2f} px | desvio y {y.std():.2f} px "
              f"| desvio combinado {desvio:.2f} px")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("desempenho", "jitter"):
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "desempenho":
        desempenho(sys.argv[2:])
    else:
        jitter(sys.argv[2])
