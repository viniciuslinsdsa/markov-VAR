import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from bcb import sgs
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.api import VAR                                     # [NOVO]
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM         # [NOVO]
warnings.simplefilter('ignore', np.exceptions.ComplexWarning)          # [NOVO] aviso interno do Johansen

# [NOVO] Padronização monetária: converte Brent e ouro (US$) para R$, para que
# todas as variáveis fiquem na mesma moeda do Ibovespa. Use False para
# reproduzir os resultados da versão anterior (Brent e ouro em US$).
PADRONIZAR_REAIS = True


# ==============================================================================
# 1. COLETA E TRATAMENTO DOS DADOS (APIs)
# ==============================================================================
print("1/7. Coletando dados via API (Yahoo Finance e Banco Central)...")

start_date = "2010-01-01"
end_date = "2024-12-31"
end_date_yf = "2025-01-01"   # o 'end' do yfinance é exclusivo

tickers = {'IBOV': '^BVSP', 'BRENT': 'BZ=F', 'OURO': 'GC=F'}

series_mensais = {}
for nome, simbolo in tickers.items():
    d = yf.download(simbolo, start=start_date, end=end_date_yf,
                    interval='1d', auto_adjust=False, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    s = d['Close'].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    series_mensais[nome] = s.resample('MS').last()
    print(f"   {nome} ({simbolo}): {s.index.min().date()} -> {s.index.max().date()} | {len(s)} pregões")

df_yahoo = pd.DataFrame(series_mensais)

# Banco Central (SGS 3695) -> último valor do mês
df_cambio = sgs.get({'CAMB': 3695}, start=start_date, end=end_date)
df_cambio.index = pd.to_datetime(df_cambio.index)
df_cambio = df_cambio.resample('MS').last()

# ------------------------------------------------------------------------------
# 1.1 Diagnóstico de meses ausentes / nulos
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print(" 1.1 DIAGNÓSTICO DE DADOS AUSENTES")
print("=" * 70)

meses_esperados = pd.date_range(start=start_date, end=end_date, freq='MS')
df_bruto_full = df_yahoo.join(df_cambio, how='outer').reindex(meses_esperados)
df_bruto_full = df_bruto_full[['IBOV', 'BRENT', 'OURO', 'CAMB']]

print(f"Meses esperados: {len(meses_esperados)}")
print("Contagem de NaN por variável (preços):")
print(df_bruto_full.isna().sum().to_string())

df_bruto = df_bruto_full.dropna()

# [NOVO] Padronização monetária (todas as séries em R$)
if PADRONIZAR_REAIS:
    df_bruto['BRENT'] = df_bruto['BRENT'] * df_bruto['CAMB']
    df_bruto['OURO'] = df_bruto['OURO'] * df_bruto['CAMB']
    print("\nPadronização monetária: Brent e ouro convertidos para R$ (× câmbio).")

# Log-retornos em PONTOS PERCENTUAIS (x100): melhora a estabilidade numérica
# da otimização. Coeficientes (elasticidades) não mudam; intercepto fica em p.p.
returns = 100 * np.log(df_bruto / df_bruto.shift(1)).dropna()
returns = returns[['IBOV', 'BRENT', 'OURO', 'CAMB']]

meses_perdidos = meses_esperados.difference(returns.index)
print(f"\nMeses efetivos no modelo: {len(returns)} | fora do modelo: "
      f"{[d.strftime('%Y-%m') for d in meses_perdidos]}")
print(f"NaN restantes em 'returns': {int(returns.isna().sum().sum())}")
gaps = returns.index.to_series().diff().dt.days.dropna()
print(f"Buracos no calendário (gap > 31 dias)? {(gaps > 31).any()}")

print("\nCorrelação entre as variáveis:")
print(returns.corr().round(3).to_string())
print()


# ==============================================================================
# 2. TESTES DE ESTACIONARIEDADE (ADF e KPSS)
# ==============================================================================
print("=" * 70)
print(" 2. TESTES DE ESTACIONARIEDADE (ADF e KPSS)")
print("=" * 70)


def executar_testes_estacionariedade(df, regressao='c'):
    resultados = []
    for col in df.columns:
        serie = df[col]
        adf_stat, adf_p, _, _, _, _ = adfuller(serie, regression=regressao, autolag='AIC')
        with warnings.catch_warnings():
            # Aviso esperado: p-valor do KPSS > 0.10 (limite da tabela)
            warnings.simplefilter('ignore', InterpolationWarning)
            kpss_stat, kpss_p, _, _ = kpss(serie, regression=regressao, nlags='auto')
        resultados.append({
            'Variável': col,
            'ADF Stat': f"{adf_stat:.4f}",
            'ADF p-valor': f"{adf_p:.4f}",
            'ADF Conclusão': 'Estacionária (p < 0.05)' if adf_p < 0.05 else 'Não Estacionária',
            'KPSS Stat': f"{kpss_stat:.4f}",
            'KPSS p-valor': f"{kpss_p:.4f}",
            'KPSS Conclusão': 'Estacionária (p > 0.05)' if kpss_p > 0.05 else 'Não Estacionária'
        })
    return pd.DataFrame(resultados)


print("Séries em LOG-RETORNO (com constante):")
print(executar_testes_estacionariedade(returns).to_string(index=False))
print("(KPSS p-valor = 0.1000 significa p >= 0.10, limite da tabela.)\n")

# [NOVO] Teste nas séries em NÍVEL (log dos preços), com constante e tendência.
# Esperado: não estacionárias em nível -> I(1), condição para testar cointegração.
niveis = np.log(df_bruto[['IBOV', 'BRENT', 'OURO', 'CAMB']])
print("Séries em LOG-NÍVEL (com constante e tendência):")
print(executar_testes_estacionariedade(niveis, regressao='ct').to_string(index=False))
print("-> Nível não estacionário + retorno estacionário = séries I(1).\n")


# ==============================================================================
# 3. SELEÇÃO DA ORDEM DE DEFASAGEM (AIC / BIC / HQ) - VIA OLS, AMOSTRA COMUM
# ==============================================================================
# Obs.: critérios de informação NÃO são confiáveis em Markov-Switching quando
# o modelo gera soluções degeneradas (variância -> 0 num regime). Por isso a
# defasagem é escolhida na regressão linear (OLS) com a mesma amostra.
print("=" * 70)
print(" 3. CRITÉRIOS DE INFORMAÇÃO PARA SELEÇÃO DE DEFASAGENS (OLS)")
print("=" * 70)

exog_cols = ['BRENT', 'OURO', 'CAMB']
max_lag = 2

X_lags = returns[exog_cols].copy()
for col in exog_cols:
    for i in range(1, max_lag + 1):
        X_lags[f'{col}_lag{i}'] = returns[col].shift(i)
X_lags = X_lags.dropna()
Y_lags = returns['IBOV'].loc[X_lags.index]

resultados_lags = []
for lag in [0, 1, 2]:
    cols = exog_cols + [f'{c}_lag{i}' for c in exog_cols for i in range(1, lag + 1)]
    res_ols = sm.OLS(Y_lags, sm.add_constant(X_lags[cols])).fit()
    resultados_lags.append({
        'Defasagens': lag, 'Obs.': int(res_ols.nobs),
        'AIC': f"{res_ols.aic:.3f}", 'BIC': f"{res_ols.bic:.3f}",
        'HQIC': f"{res_ols.info_criteria('hqic'):.3f}"
    })
print(pd.DataFrame(resultados_lags).to_string(index=False))
print("-> Menor critério = melhor especificação (amostra idêntica nos 3 modelos).\n")


# ==============================================================================
# 4. ESTIMAÇÃO DO MODELO MARKOV-SWITCHING (2 REGIMES) COM MULTI-START ROBUSTO
# ==============================================================================
print("=" * 70)
print(" 4. ESTIMATIVA DO MODELO DE MUDANÇA DE REGIME (MARKOV-SWITCHING)")
print("=" * 70)

Y = returns['IBOV']
X = returns[exog_cols]
beta_ols = sm.OLS(Y, sm.add_constant(X)).fit().params.values   # [const, b1, b2, b3]

# Critérios para descartar soluções DEGENERADAS (variância ~ 0 ou regime com
# pouquíssimas observações, onde o modelo "memoriza" alguns meses)
MIN_SIGMA2 = 1.0     # variância mínima por regime, em (p.p.)^2  (desvio >= 1 p.p.)
MIN_SHARE = 0.10     # cada regime deve ter >= 10% da amostra (prob. suavizada média)
N_STARTS = 80


def nomes_base(nomes):
    """Nomes-base dos parâmetros de regressão, na ordem: const, x1, x2, x3."""
    base = []
    for n in nomes:
        if '->' in n or n.startswith('sigma2'):
            continue
        b = n.split('[')[0]
        if b not in base:
            base.append(b)
    return base


def gerar_start(model, rng):
    nomes = list(model.param_names)
    base = nomes_base(nomes)
    var_y = float(Y.var())
    inverte = rng.random() < 0.5
    sp = np.zeros(len(nomes))
    for j, n in enumerate(nomes):
        if n == 'p[0->0]':
            sp[j] = rng.uniform(0.75, 0.97)
        elif n == 'p[1->0]':
            sp[j] = rng.uniform(0.03, 0.25)
        elif n.startswith('sigma2'):
            reg = int(n.split('[')[1].rstrip(']'))
            baixa, alta = var_y * rng.uniform(0.2, 0.6), var_y * rng.uniform(1.5, 3.0)
            sp[j] = (alta if inverte else baixa) if reg == 0 else (baixa if inverte else alta)
        else:
            k = base.index(n.split('[')[0])
            escala = 0.5 if k == 0 else 0.15
            sp[j] = beta_ols[k] + rng.normal(0, escala)
    return sp


def regime_valido(res):
    if not np.all(np.isfinite(res.params.values)) or not np.isfinite(res.llf):
        return False
    sig2 = [res.params[f'sigma2[{i}]'] for i in (0, 1)]
    if min(sig2) < MIN_SIGMA2:
        return False
    share = res.smoothed_marginal_probabilities.mean(axis=0)
    return float(share.min()) >= MIN_SHARE


def estimar_multistart(model, rotulo, seed):
    rng = np.random.default_rng(seed)
    melhor, n_ok = None, 0
    for _ in range(N_STARTS):
        sp = gerar_start(model, rng)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = model.fit(start_params=sp, em_iter=0, maxiter=300, disp=False)
            if not regime_valido(r):
                continue
        except Exception:
            continue
        n_ok += 1
        if melhor is None or r.llf > melhor.llf:
            melhor = r
    print(f"[{rotulo}] soluções válidas (não degeneradas): {n_ok}/{N_STARTS}")
    if melhor is None:
        raise SystemExit(f"[{rotulo}] Nenhuma solução não degenerada encontrada. "
                         "Reduza o número de coeficientes que mudam por regime "
                         "(ex.: switching_exog=[False, False, True]) ou use o modelo de robustez.")
    return melhor


# Modelo completo: intercepto, coeficientes e variância mudam por regime
ms_model = MarkovRegression(endog=Y, exog=X, k_regimes=2,
                            switching_trend=True, switching_exog=True,
                            switching_variance=True)
fit_ms = estimar_multistart(ms_model, 'Modelo completo', seed=123)
print(fit_ms.summary())

# Robustez: só intercepto e variância mudam (coeficientes fixos)
ms_vol = MarkovRegression(endog=Y, exog=X, k_regimes=2,
                          switching_trend=True, switching_exog=False,
                          switching_variance=True)
fit_vol = estimar_multistart(ms_vol, 'Modelo de robustez', seed=456)

print("\nComparação entre especificações:")
comp = pd.DataFrame({
    'Modelo': ['Completo (const, coef., variância mudam)', 'Robustez (const e variância mudam)'],
    'Log-Likelihood': [fit_ms.llf, fit_vol.llf],
    'AIC': [fit_ms.aic, fit_vol.aic],
    'BIC': [fit_ms.bic, fit_vol.bic],
})
print(comp.round(3).to_string(index=False))

# Teste da razão de verossimilhança: coeficientes iguais entre regimes (3 restrições)
LR = 2 * (fit_ms.llf - fit_vol.llf)
print(f"\nTeste LR (H0: coeficientes de BRENT, OURO e CAMB iguais nos dois regimes): "
      f"LR = {LR:.3f} | p-valor (qui², 3 g.l.) = {1 - chi2.cdf(max(LR, 0), 3):.4f}")
print("(Aproximação assintótica; interpretar com cautela em amostra de 179 meses.)\n")


# ==============================================================================
# 5. DIAGNÓSTICO DOS RESÍDUOS (LJUNG-BOX)
# ==============================================================================
print("=" * 70)
print(" 5. DIAGNÓSTICO DOS RESÍDUOS PADRONIZADOS (LJUNG-BOX)")
print("=" * 70)

nomes_ms = nomes_base(list(fit_ms.params.index))          # ex.: ['const','x1','x2','x3']
mapa = dict(zip(['const'] + exog_cols, nomes_ms))
print(f"Mapeamento de nomes dos parâmetros: {mapa}")

P = fit_ms.smoothed_marginal_probabilities
mu = np.zeros(len(Y))
var = np.zeros(len(Y))
for r in (0, 1):
    xb = fit_ms.params[f"{mapa['const']}[{r}]"] + sum(
        fit_ms.params[f"{mapa[v]}[{r}]"] * X[v].values for v in exog_cols)
    mu += P[r].values * xb
    var += P[r].values * fit_ms.params[f'sigma2[{r}]']
z = pd.Series((Y.values - mu) / np.sqrt(var), index=Y.index)

print("\nResíduos padronizados:")
print(acorr_ljungbox(z, lags=[6, 12], return_df=True))
print("\nResíduos padronizados ao quadrado (efeitos ARCH):")
print(acorr_ljungbox(z ** 2, lags=[6, 12], return_df=True))
print("-> p-valores > 0.05 indicam ausência de autocorrelação.\n")


# ==============================================================================
# 6. IDENTIFICAÇÃO DOS REGIMES, DURAÇÃO E TESTES DE IGUALDADE DOS COEFICIENTES
# ==============================================================================
print("=" * 70)
print(" 6. PROPRIEDADES DOS REGIMES")
print("=" * 70)

p00 = fit_ms.params['p[0->0]']
p10 = fit_ms.params['p[1->0]']
p11 = 1 - p10
duracoes_est = {0: 1 / (1 - p00), 1: 1 / (1 - p11)}
sigma2_est = {0: fit_ms.params['sigma2[0]'], 1: fit_ms.params['sigma2[1]']}   # variância RESIDUAL

# Convenção de numeração (apenas um rótulo, sem significado econômico):
# Regime 0 = menor variância residual | Regime 1 = maior variância residual.
ordem = sorted(sigma2_est, key=sigma2_est.get)
idx_r0, idx_r1 = ordem[0], ordem[1]
rotulo = {idx_r0: 'Regime 0', idx_r1: 'Regime 1'}

prob_r0 = P[idx_r0]
prob_r1 = P[idx_r1]

print("Convenção: Regime 0 = menor variância residual; Regime 1 = maior variância residual.\n")
for r in (idx_r0, idx_r1):
    m = P[r] > 0.5
    print(f"{rotulo[r]}: sigma2 residual = {sigma2_est[r]:.3f} | duração esperada = {duracoes_est[r]:.2f} meses "
          f"| meses (prob. > 0.5) = {int(m.sum())} | prob. suavizada média = {P[r].mean():.3f}")
    print(f"           IBOV nos meses do regime: média = {Y[m].mean():+.2f} p.p. | desvio-padrão = {Y[m].std():.2f} p.p.")


def episodios(mask):
    """Lista os episódios contínuos em que mask é verdadeira."""
    grupo = (mask != mask.shift()).cumsum()
    saida = []
    for _, g in mask.groupby(grupo):
        if g.iloc[0]:
            saida.append((g.index[0].strftime('%Y-%m'), g.index[-1].strftime('%Y-%m'), len(g)))
    return saida


for r in (idx_r0, idx_r1):
    print(f"\nEpisódios contínuos no {rotulo[r]}:")
    for ini, fim, n in episodios(P[r] > 0.5):
        print(f"   {ini} -> {fim} ({n} meses)")

print("\nProbabilidade suavizada do Regime 0 em meses selecionados:")
for mes in ['2011-08', '2015-09', '2018-05', '2020-03', '2022-10']:
    ts = pd.Timestamp(mes + '-01')
    if ts in prob_r0.index:
        print(f"  {mes}: {prob_r0.loc[ts]:.3f}")

# ------------------------------------------------------------------------------
# Coeficientes por regime e teste de Wald de igualdade entre regimes
# ------------------------------------------------------------------------------
params = fit_ms.params
cov = np.asarray(fit_ms.cov_params())
nomes_params = list(params.index)


def wald_igualdade(variaveis):
    R = np.zeros((len(variaveis), len(params)))
    for i, v in enumerate(variaveis):
        R[i, nomes_params.index(f'{mapa[v]}[0]')] = 1
        R[i, nomes_params.index(f'{mapa[v]}[1]')] = -1
    d = R @ params.values
    try:
        W = float(d @ np.linalg.solve(R @ cov @ R.T, d))
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    return W, float(1 - chi2.cdf(W, len(variaveis)))


print("\n" + "-" * 70)
print(" COEFICIENTES POR REGIME (erro-padrão entre parênteses) E TESTE DE WALD")
print("-" * 70)
linhas = []
for v in ['const'] + exog_cols:
    W, pv = wald_igualdade([v])
    linhas.append({
        'Variável': v,
        'Regime 0': f"{params[f'{mapa[v]}[{idx_r0}]']:+.4f} ({fit_ms.bse[f'{mapa[v]}[{idx_r0}]']:.4f})",
        'Regime 1': f"{params[f'{mapa[v]}[{idx_r1}]']:+.4f} ({fit_ms.bse[f'{mapa[v]}[{idx_r1}]']:.4f})",
        'Wald (qui²)': f"{W:.3f}",
        'p-valor': f"{pv:.4f}",
        'Difere (5%)?': 'Sim' if pv < 0.05 else ('Não' if np.isfinite(pv) else 'n/d')
    })
print(pd.DataFrame(linhas).to_string(index=False))

W_c, p_c = wald_igualdade(exog_cols)
print(f"\nTeste conjunto (BRENT, OURO e CAMB iguais nos dois regimes): "
      f"qui²({len(exog_cols)}) = {W_c:.3f} | p-valor = {p_c:.4f}")
print("-> 'n/d' ou nan: matriz de covariância singular; os erros-padrão não são confiáveis.\n")

# Exporta as probabilidades suavizadas para uso no texto/tabelas
pd.DataFrame({'prob_regime0': prob_r0, 'prob_regime1': prob_r1}).to_csv('probabilidades_regimes.csv')


# ==============================================================================
# 7. GRÁFICO DAS PROBABILIDADES SUAVIZADAS
# ==============================================================================
REGIME_PLOT = 0   # regime cuja probabilidade será plotada (0 ou 1)

prob_sel = prob_r1 if REGIME_PLOT == 1 else prob_r0
calendario = pd.date_range(returns.index.min(), returns.index.max(), freq='MS')
prob_plot = prob_sel.reindex(calendario)
print(f"Meses faltantes no gráfico: {int(prob_plot.isna().sum())}")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(prob_plot.index, prob_plot.values, color='firebrick', linewidth=1.5,
        label=f'Probabilidade suavizada do Regime {REGIME_PLOT}')
ax.fill_between(prob_plot.index, 0, 1, where=(prob_plot.fillna(0).values > 0.5),
                color='firebrick', alpha=0.10, label=f'Meses classificados no Regime {REGIME_PLOT}')
ax.axhline(0.5, color='black', linestyle='--', alpha=0.7, label='Limiar de Decisão (50%)')

ax.set_title(f'Probabilidade Suavizada do Regime {REGIME_PLOT} no Mercado Acionário Brasileiro (2010–2024)',
             fontsize=12, fontweight='bold')
ax.set_ylabel(f'Probabilidade Latente $P(S_t = {REGIME_PLOT})$')
ax.set_xlabel('Ano')
ax.set_ylim(-0.02, 1.02)
ax.set_xlim(prob_plot.index.min(), prob_plot.index.max())
ax.grid(True, alpha=0.3)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

fig.tight_layout()
fig.savefig(f'probabilidade_regime{REGIME_PLOT}.png', dpi=200, bbox_inches='tight')
plt.show()

# ==============================================================================
# 8. [NOVO] COINTEGRAÇÃO DE JOHANSEN (séries em log-nível)
# ==============================================================================
print("=" * 70)
print(" 8. TESTE DE COINTEGRAÇÃO DE JOHANSEN")
print("=" * 70)

ordem_var = ['BRENT', 'OURO', 'CAMB', 'IBOV']   # ordem de Cholesky: do mais exógeno ao IBOV
niveis = niveis[ordem_var]

# Defasagens do VAR em nível (BIC); o Johansen usa essa ordem menos 1 (em diferenças)
sel_nivel = VAR(niveis).select_order(maxlags=6, trend='c')
p_nivel = max(sel_nivel.selected_orders['bic'], 1)
k_diff = p_nivel - 1
print(f"Defasagens do VAR em nível (BIC): {p_nivel} -> Johansen com {k_diff} defasagem(ns) em diferença")

jo = coint_johansen(niveis, det_order=0, k_ar_diff=k_diff)
tab_jo = pd.DataFrame({
    'H0: posto <=': range(len(ordem_var)),
    'Traço': jo.lr1.round(3), 'VC 5% (traço)': jo.cvt[:, 1],
    'Máx. autovalor': jo.lr2.round(3), 'VC 5% (máx.)': jo.cvm[:, 1],
})
print(tab_jo.to_string(index=False))

# Posto = primeiro r cuja H0 não é rejeitada pelo teste do traço
posto = 0
while posto < len(ordem_var) and jo.lr1[posto] > jo.cvt[posto, 1]:
    posto += 1
print(f"\nPosto de cointegração (traço, 5%): {posto}")
if posto == 0:
    print("-> Sem cointegração: estima-se um VAR nos log-retornos.\n")
else:
    print("-> Há cointegração: estima-se um VEC (VECM) nas séries em log-nível.\n")


# ==============================================================================
# 9. [NOVO] SELEÇÃO DE DEFASAGENS DO VAR (AIC / BIC / HQ / FPE)
# ==============================================================================
print("=" * 70)
print(" 9. SELEÇÃO DE DEFASAGENS DO VAR")
print("=" * 70)

sel = VAR(returns[ordem_var]).select_order(maxlags=6, trend='c')
print(sel.summary())
print(f"Ordens selecionadas: { {k: int(v) for k, v in sel.selected_orders.items()} }")
p_var = max(sel.selected_orders['bic'], 1)
print(f"Defasagens utilizadas: {p_var} (BIC; mínimo de 1 para haver dinâmica no VAR)\n")


# ==============================================================================
# 10. [NOVO] ESTIMAÇÃO DO VAR OU DO VEC, RELAÇÕES CONTEMPORÂNEAS,
#     IMPULSO-RESPOSTA E DECOMPOSIÇÃO DA VARIÂNCIA
# ==============================================================================
print("=" * 70)
print(" 10. ESTIMAÇÃO DO MODELO " + ("VAR" if posto == 0 else "VEC"))
print("=" * 70)

if posto == 0:
    modelo = VAR(returns[ordem_var]).fit(p_var, trend='c')
    print(f"VAR({p_var}) nos log-retornos — equação do IBOV:")
    print(pd.DataFrame({'Coef.': modelo.params['IBOV'], 'p-valor': modelo.pvalues['IBOV']}).round(4))
    print(f"\nEstável? {modelo.is_stable()}")
    print(f"Autocorrelação dos resíduos (Portmanteau, 12 def.): "
          f"p = {modelo.test_whiteness(nlags=12, adjusted=True).pvalue:.4f}")
    residuos = modelo.resid
else:
    modelo = VECM(niveis, k_ar_diff=k_diff, coint_rank=posto, deterministic='ci').fit()
    print("Vetor de cointegração (beta) — relação de longo prazo:")
    print(pd.DataFrame(modelo.beta, index=ordem_var).round(4))
    print("\nVelocidade de ajuste (alfa) e p-valores:")
    print(pd.DataFrame({'alfa': modelo.alpha[:, 0], 'p-valor': modelo.pvalues_alpha[:, 0]},
                       index=ordem_var).round(4))
    print(f"\nAutocorrelação dos resíduos (Portmanteau, 12 def.): "
          f"p = {modelo.test_whiteness(nlags=12, adjusted=True).pvalue:.4f}")
    residuos = pd.DataFrame(modelo.resid, columns=ordem_var)

# Relações contemporâneas: correlação dos resíduos e causalidade instantânea
print("\nRelações contemporâneas (correlação dos resíduos):")
print(residuos.corr().round(3).to_string())
print("\nCausalidade instantânea e de Granger com o IBOV (p-valores):")
for v in ['BRENT', 'OURO', 'CAMB']:
    if posto == 0:
        inst = modelo.test_inst_causality([v]).pvalue
        gran = modelo.test_causality('IBOV', [v], kind='wald').pvalue
    else:
        inst = modelo.test_inst_causality(causing=v).pvalue
        gran = modelo.test_granger_causality(caused='IBOV', causing=v).pvalue
    print(f"  {v}: instantânea p = {inst:.4f} | {v} -> IBOV (Granger) p = {gran:.4f}")

# Impulso-resposta (Cholesky, ordem: BRENT, OURO, CAMB, IBOV)
irf = modelo.irf(12)
if posto == 0:
    irf.plot(orth=True, response='IBOV')          # com intervalo de confiança de 95%
    choques = irf.orth_irfs
else:
    irf.plot(orth=True, response='IBOV', plot_stderr=False)
    choques = modelo.orth_ma_rep(12)
plt.suptitle('Resposta do IBOV a choques de 1 desvio-padrão (Cholesky)')
plt.show()

i_ibov = ordem_var.index('IBOV')
print("\nResposta do IBOV a choques ortogonalizados (meses 0, 1, 2, 3, 6, 12):")
print(pd.DataFrame({v: choques[[0, 1, 2, 3, 6, 12], i_ibov, ordem_var.index(v)]
                    for v in ['BRENT', 'OURO', 'CAMB']},
                   index=[0, 1, 2, 3, 6, 12]).round(4))

# Decomposição da variância do erro de previsão do IBOV
acum = np.cumsum(choques ** 2, axis=0)
fevd_ibov = acum[:, i_ibov, :] / acum[:, i_ibov, :].sum(axis=1, keepdims=True)
print("\nDecomposição da variância do IBOV (%):")
print(pd.DataFrame(100 * fevd_ibov[[0, 2, 5, 11]], index=[1, 3, 6, 12],
                   columns=ordem_var).round(2))
print()


# ==============================================================================
# 11. [NOVO] DUMMY DE PANDEMIA (quebra estrutural)
# ==============================================================================
print("=" * 70)
print(" 11. DUMMY DE PANDEMIA")
print("=" * 70)

INICIO_DUMMY, FIM_DUMMY = '2020-03-01', '2020-12-01'
dummy = ((returns.index >= INICIO_DUMMY) & (returns.index <= FIM_DUMMY)).astype(float)
print(f"Dummy = 1 de {INICIO_DUMMY[:7]} a {FIM_DUMMY[:7]} ({int(dummy.sum())} meses)")

# (a) Dummy no modelo VAR/VEC: a pandemia deslocou o nível das variáveis?
if posto == 0:
    m_d = VAR(returns[ordem_var], exog=pd.DataFrame({'DUMMY': dummy}, index=returns.index)).fit(p_var, trend='c')
    coef_d, p_d = m_d.params.loc['DUMMY'], m_d.pvalues.loc['DUMMY']
else:
    d_nivel = ((niveis.index >= INICIO_DUMMY) & (niveis.index <= FIM_DUMMY)).astype(float)
    m_d = VECM(niveis, exog=d_nivel[:, None], k_ar_diff=k_diff, coint_rank=posto,
               deterministic='ci').fit()
    coef_d = pd.Series(m_d.det_coef[:, -1], index=ordem_var)
    p_d = pd.Series(m_d.pvalues_det_coef[:, -1], index=ordem_var)
print("\n(a) Coeficiente da dummy em cada equação do modelo:")
print(pd.DataFrame({'Coef.': coef_d, 'p-valor': p_d}).round(4))

# (b) Interação: a relação do IBOV com petróleo, ouro e câmbio mudou na pandemia?
X_d = returns[exog_cols].copy()
X_d['DUMMY'] = dummy
for v in exog_cols:
    X_d[f'{v}_x_DUMMY'] = returns[v] * dummy
ols_d = sm.OLS(returns['IBOV'], sm.add_constant(X_d)).fit()
print("\n(b) Regressão do IBOV com interações (teste do tipo Chow):")
print(pd.DataFrame({'Coef.': ols_d.params, 'p-valor': ols_d.pvalues}).round(4))
teste = ols_d.f_test(' , '.join(f'{v}_x_DUMMY = 0' for v in exog_cols))
print(f"\nTeste conjunto (interações = 0): F = {float(np.squeeze(teste.fvalue)):.3f} | "
      f"p-valor = {float(teste.pvalue):.4f}")
print("-> p < 0,05: a relação contemporânea do IBOV com as variáveis mudou na pandemia.")
print("   Obs.: a amostra não é dividida em pré/pandemia/pós porque a subamostra da")
print("   pandemia teria poucas observações para um modelo com 4 variáveis.")