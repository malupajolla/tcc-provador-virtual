 Provador Virtual de Acessórios Faciais

Sistema de prova virtual em tempo real de acessórios faciais (óculos, chapéus, brincos e piercings) usando a webcam. O rosto é rastreado com MediaPipe Face Mesh e os acessórios são sobrepostos ao vídeo com OpenCV, acompanhando os movimentos da cabeça.

Projeto desenvolvido como Trabalho de Conclusão de Curso (TCC) em Ciência da Computação na UNIP – Ribeirão Preto.

<img width="605" height="353" alt="image" src="https://github.com/user-attachments/assets/4721b31b-9cca-4bc7-8c16-567e62137d1d" />

Funcionalidades:
-Detecção facial em tempo real com 468 pontos de referência (MediaPipe Face Mesh)
-Sobreposição de acessórios em PNG com transparência (canal alpha)
-Várias categorias de acessórios: óculos, chapéus, brincos e piercings
-Navegação por categoria e troca de acessório
-Suavização de movimento para reduzir tremidas no encaixe
-Privacidade em primeiro lugar: nenhuma imagem ou dado facial é salvo, exportado ou transmitido, tudo acontece só na tela, em tempo real

Tecnologias:
Python 3	Linguagem principal
OpenCV	Captura da webcam, processamento e exibição das imagens
MediaPipe Face Mesh	Detecção dos pontos do rosto
NumPy	Operações com as imagens

Como funciona?
A webcam captura cada quadro do vídeo.
O MediaPipe Face Mesh localiza os pontos do rosto.
Pontos-chave definem a posição de cada tipo de acessório — por exemplo, os cantos dos olhos (pontos 33 e 263) para os óculos e a testa (ponto 10) para o chapéu.
A distância e o ângulo entre esses pontos determinam a escala e a rotação do acessório.
O PNG do acessório é mesclado ao quadro usando o canal de transparência.

Como executar?
Pré-requisitos:
Python 3.10+ <!-- AJUSTAR: versão que você usa -->
Uma webcam
Instalação
bash
git clone https://github.com/malupajolla/tcc-provador-virtual.git
cd tcc-provador-virtual
python -m venv venv
venv\Scripts\activate        # Windows

pip install -r requirements.txt
Execução
bash
python main.py   # AJUSTAR: nome do seu script
Controles <!-- AJUSTAR: confira se ainda estão assim -->
Tecla	Ação
1–4	Trocar categoria
A / D	Acessório anterior / próximo
Q	Sair

📁 Estrutura do projeto <!-- AJUSTAR -->
├── main.py
├── accessories/
│   ├── glasses/
│   ├── hats/
│   ├── earrings/
│   └── piercings/
├── requirements.txt
└── README.md

🗺️ Próximos passos
 Encaixe automático aprimorado dos acessórios, sem ajuste manual
 Interface web no estilo "espelho" 
 Contorno de guia quando o rosto não é detectado


Maria Luiza — @malupajolla
