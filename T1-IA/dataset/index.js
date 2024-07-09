const fs = require("fs");

const file = fs.readFileSync("./tic-tac-toe.data").toString();
const lines = file
  .split("\n")
  .filter((l) => l.endsWith("win") || l.endsWith("lost"));

const partidas = [];

/* ----- FUNÇÕES DE CHECAGEM DE VITÓRIA ----- */

const verificarVitoria = (tabuleiro, indices, vencedor) => 
  indices.every((i) => tabuleiro[i] === vencedor);

const vitoriaNaDiagonal = (tabuleiro, vencedor) =>
  verificarVitoria(tabuleiro, [0, 4, 8], vencedor) || verificarVitoria(tabuleiro, [2, 4, 6], vencedor);

const vitoriaNaLinha = (tabuleiro, vencedor) => 
  verificarVitoria(tabuleiro, [0, 1, 2], vencedor) ||
  verificarVitoria(tabuleiro, [3, 4, 5], vencedor) ||
  verificarVitoria(tabuleiro, [6, 7, 8], vencedor);

const vitoriaNaColuna = (tabuleiro, vencedor) => 
  verificarVitoria(tabuleiro, [0, 3, 6], vencedor) ||
  verificarVitoria(tabuleiro, [1, 4, 7], vencedor) ||
  verificarVitoria(tabuleiro, [2, 5, 8], vencedor);

/* ----- FUNÇÕES PARA GERAR JOGOS COM RESULTADOS ----- */

const criarPartidasComVitoria = (tabuleiro, vencedor, tipoVitoria) => {
  if (vitoriaNaLinha(tabuleiro, vencedor) || vitoriaNaColuna(tabuleiro, vencedor) || vitoriaNaDiagonal(tabuleiro, vencedor)) {
    const vitorias = tipoVitoria(tabuleiro, vencedor);
    vitorias.forEach(vitoria => partidas.push(vitoria));
  }
};

const criarPartidasPorLinha = (tabuleiro, vencedor) => {
  const [a1, a2, a3, b1, b2, b3, c1, c2, c3] = tabuleiro;
  const linhas = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8]
  ];

  linhas.forEach((linha, idx) => {
    if (verificarVitoria(tabuleiro, linha, vencedor)) {
      const novoTabuleiro = [...tabuleiro];
      linha.forEach(i => novoTabuleiro[i] = 'b');
      partidas.push(novoTabuleiro);
    }
  });
};

const criarPartidasPorColuna = (tabuleiro, vencedor) => {
  const [a1, a2, a3, b1, b2, b3, c1, c2, c3] = tabuleiro;
  const colunas = [
    [0, 3, 6], [1, 4, 7], [2, 5, 8]
  ];

  colunas.forEach((coluna, idx) => {
    if (verificarVitoria(tabuleiro, coluna, vencedor)) {
      const novoTabuleiro = [...tabuleiro];
      coluna.forEach(i => novoTabuleiro[i] = 'b');
      partidas.push(novoTabuleiro);
    }
  });
};

/* ----- PROCESSAMENTO DAS LINHAS ----- */

lines.forEach((line) => {
  const resultados = line.split(",");
  const resultado = resultados.pop();
  const tabuleiro = resultados;
  const vencedor = resultado === "win" ? "x" : "o";

  criarPartidasComVitoria(tabuleiro, vencedor, criarPartidasPorLinha);
  criarPartidasComVitoria(tabuleiro, vencedor, criarPartidasPorColuna);
});

console.log(partidas.length + " casos de jogo identificados, salvando em out.data");

const conteudo = partidas.map((tabuleiro) => tabuleiro.join(",") + ",game");
fs.writeFileSync("./out.data", conteudo.join("\n"));
