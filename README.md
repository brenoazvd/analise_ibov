# Previsão Direcional do IBOVESPA (D+1)

## 1.Objetivo:

Prever se  fechamento do IBOVESPA do **Próximo dia (d+1)** será **maior ↑**  ou **menor ↓** do que o fechamento de hoje.

**Meta do desafio:** Atingir pelo menos 75% de acurácia nos dados de teste, que devem ser os últimos 30 dias


## 2. Aquisiçao e Entendimento dos Dados

### Fonte: https://br.investing.com/indices/bovespa-historical-data
* Histórico de dados do IBOVESPA:
  * Data
  * Abertura
  * Máxima
  * Mínima
  * Último (Fechamento)
  * Var%(Variação percentual diária)
  * Vol.(Volume) 

Foram considerados o período de **2022-10-11 à 2025-10-23**

### Entendimento Inicial da fonte de dados:



* Quantidade de colunas e linhas (959, 7)

* Quantidade total de nulos: 0

* Informações do DataFrame:

  <table class="info-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Coluna</th>
                    <th>Valores Não Nulos</th>
                    <th>Tipo de Dados</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>0</td>
                    <td>Data</td>
                    <td>959 non-null</td>
                    <td>object</td>
                </tr>
                <tr>
                    <td>1</td>
                    <td>Último</td>
                    <td>959 non-null</td>
                    <td>float64</td>
                </tr>
                <tr>
                    <td>2</td>
                    <td>Abertura</td>
                    <td>959 non-null</td>
                    <td>float64</td>
                </tr>
                <tr>
                    <td>3</td>
                    <td>Máxima</td>
                    <td>959 non-null</td>
                    <td>float64</td>
                </tr>
                <tr>
                    <td>4</td>
                    <td>Mínima</td>
                    <td>959 non-null</td>
                    <td>float64</td>
                </tr>
                <tr>
                    <td>5</td>
                    <td>Vol.</td>
                    <td>959 non-null</td>
                    <td>object</td>
                </tr>
                <tr>
                    <td>6</td>
                    <td>Var%</td>
                    <td>959 non-null</td>
                    <td>object</td>
                </tr>
            </tbody>
        </table>
      

    


* Estatísticas descritivas do DataFrame:


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Último</th>
      <td>959.0</td>
      <td>120.914576</td>
      <td>11.700480</td>
      <td>96.121</td>
      <td>111.5440</td>
      <td>120.586</td>
      <td>129.6010</td>
      <td>146.492</td>
    </tr>
    <tr>
      <th>Abertura</th>
      <td>959.0</td>
      <td>120.871945</td>
      <td>11.683371</td>
      <td>96.119</td>
      <td>111.4645</td>
      <td>120.561</td>
      <td>129.5280</td>
      <td>146.492</td>
    </tr>
    <tr>
      <th>Máxima</th>
      <td>959.0</td>
      <td>121.767710</td>
      <td>11.614397</td>
      <td>96.971</td>
      <td>112.5285</td>
      <td>121.607</td>
      <td>130.1425</td>
      <td>147.578</td>
    </tr>
    <tr>
      <th>Mínima</th>
      <td>959.0</td>
      <td>120.038944</td>
      <td>11.796513</td>
      <td>95.267</td>
      <td>110.3985</td>
      <td>120.022</td>
      <td>128.7335</td>
      <td>146.067</td>
    </tr>
  </tbody>
</table>
</div>





### Tratamentos Iniciais

* Conversão de `Data` para Datetime:

    ```python
          df['Data']=pd.to_datetime(df['Data'],format='%d.%m.%Y')
    
   ```
* Conversão de `Vol.` para numero:

     ```python
      def converter_valor_volume(x):
      try:
          s = str(x).strip().upper().replace(',', '.')
          if s.endswith('M'):
              return float(s[:-1]) * 1_000_000
          elif s.endswith('B'):
              return float(s[:-1]) * 1_000_000_000
          else:
              return float(s)
      except:
          return None

     df['Vol.'] = df['Vol.'].apply(converter_valor_volume)
    ```
    
* Conversão `Var%` de string com % para float
  ```python
    def converter_percentual(x):

    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace('%','').replace('.','').replace(',','.')
    try:
        return float(s)/100.0
    except:
        return np.nan

  df['Var%'] = df['Var%'].apply(converter_percentual)
  ```

* Criação de colunas de datas
```python
   df['ano']=df['Data'].dt.year
   df['mes']=df['Data'].dt.month
   df['day_of_week']=df['Data'].dt.day_of_week
   df['Anomes']=df['Data'].dt.strftime('%m.%y')
  ```


### Exploração dos dados

* Visualização descritiva do dataframe após correção de variaveis


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Data</th>
      <td>959</td>
      <td>2023-11-26 14:57:56.120959232</td>
      <td>2021-12-27 00:00:00</td>
      <td>2022-12-10 12:00:00</td>
      <td>2023-11-27 00:00:00</td>
      <td>2024-11-06 12:00:00</td>
      <td>2025-10-24 00:00:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Último</th>
      <td>959.0</td>
      <td>120.914576</td>
      <td>96.121</td>
      <td>111.544</td>
      <td>120.586</td>
      <td>129.601</td>
      <td>146.492</td>
      <td>11.70048</td>
    </tr>
    <tr>
      <th>Abertura</th>
      <td>959.0</td>
      <td>120.871945</td>
      <td>96.119</td>
      <td>111.4645</td>
      <td>120.561</td>
      <td>129.528</td>
      <td>146.492</td>
      <td>11.683371</td>
    </tr>
    <tr>
      <th>Máxima</th>
      <td>959.0</td>
      <td>121.76771</td>
      <td>96.971</td>
      <td>112.5285</td>
      <td>121.607</td>
      <td>130.1425</td>
      <td>147.578</td>
      <td>11.614397</td>
    </tr>
    <tr>
      <th>Mínima</th>
      <td>959.0</td>
      <td>120.038944</td>
      <td>95.267</td>
      <td>110.3985</td>
      <td>120.022</td>
      <td>128.7335</td>
      <td>146.067</td>
      <td>11.796513</td>
    </tr>
    <tr>
      <th>Vol.</th>
      <td>959.0</td>
      <td>1981453514.077164</td>
      <td>4330000.0</td>
      <td>9945000.0</td>
      <td>12310000.0</td>
      <td>17455000.0</td>
      <td>24870000000.0</td>
      <td>3981848092.265336</td>
    </tr>
    <tr>
      <th>Var%</th>
      <td>959.0</td>
      <td>0.000402</td>
      <td>-0.0335</td>
      <td>-0.0058</td>
      <td>0.0003</td>
      <td>0.0069</td>
      <td>0.0554</td>
      <td>0.010604</td>
    </tr>
    <tr>
      <th>ano</th>
      <td>959.0</td>
      <td>2023.422315</td>
      <td>2021.0</td>
      <td>2022.0</td>
      <td>2023.0</td>
      <td>2024.0</td>
      <td>2025.0</td>
      <td>1.105104</td>
    </tr>
    <tr>
      <th>mes</th>
      <td>959.0</td>
      <td>6.296142</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>12.0</td>
      <td>3.345126</td>
    </tr>
    <tr>
      <th>dow</th>
      <td>959.0</td>
      <td>1.995829</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.414207</td>
    </tr>
    <tr>
      <th>day_of_week</th>
      <td>959.0</td>
      <td>1.995829</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.414207</td>
    </tr>
  </tbody>
</table>



` Identificamos movimento irregular na variavel de volume, com grande valor de desvio padrão `

### * Entendendo melhor as variáveis

* <img width="990" height="790" alt="image" src="https://github.com/user-attachments/assets/94f2800d-3345-42d2-925c-6082404ffc1c" />


  * **1.Distribuição por Volume de Negócios**: Valores com alguma irregularidade reforçando o que notamos anteriormente olhando o desvio padrão 
  * **2.Distribuição de Variação Percentual (%)**: Podemos notar uma distribuição normal dos dados
  * **3.Distribuição de Último Preço**: Também notamos uma distribuição normal do preço, sem grandes outliers
  * **4.Correlação de variaveis com Último Preço**: Nota-se uma correlação muito forte entre as variaveis de valores (Abertura, Minima, Máxima)

 
 


### Engenharia de atributos

  
#Função para criação de features
```python
def create_features(df,date_col,close_col,vol_col,windows=(5,10,20,50,100)):
    """Cria features técnicas para análise de séries temporais financeiras.
    Parâmetros:
    df: DataFrame contendo os dados financeiros.
    date_col: Nome da coluna de datas.
    close_col: Nome da coluna de preços de fechamento.
    vol_col: Nome da coluna de volume.
    windows: Tupla de janelas para cálculo de médias móveis e outros indicadores.
    Retorna:
    DataFrame com as novas features criadas."""


    df = df.copy()

    # Ordenando os dados pela data
    df = df.sort_values(date_col).reset_index(drop=True)

    # Calculos simples de retornos para o periodo de 1, 5 e 10 dias
    df['ret_1'] = df[close_col].pct_change(1) # retorno diário

    df["ret_5d"]  = df[close_col].pct_change(5)
    df["ret_10d"] = df[close_col].pct_change(10)


    # médias e EMAs multijanelas
    for w in windows:
        df[f'mm_{w}']  = df[close_col].rolling(window=w, min_periods=w).mean()
        df[f'ema_{w}'] = df[close_col].ewm(span=w, adjust=False).mean()
        # spread preço - média
        df[f'spread_mm{w}'] = df[close_col] - df[f'mm_{w}']
        # MÉDIA de volume
        
        df[f'volm_{w}'] = df[vol_col].rolling(window=w, min_periods=w).mean()

    # relações preço vs média (com eps para evitar div/0)
    eps = 1e-9
    for w in windows:
        df[f'close_over_mm{w}'] = (df[close_col] / (df[f'mm_{w}'] + eps)) - 1

    # Bandas de Bollinger (20 dias, 2 desvios)
    w=20
    std = df[close_col].rolling(w).std()
    ma=df['mm_20']
    df["bb_width"] = ((ma + 2*std) - (ma - 2*std)) / ma

    #Função para cálculo do RSI
    def compute_rsi(series, window=14):
        delta = series.diff()
        up = np.where(delta > 0, delta, 0.0)
        down = np.where(delta < 0, -delta, 0.0)
        roll_up = pd.Series(up, index=series.index).rolling(window).mean()
        roll_down = pd.Series(down, index=series.index).rolling(window).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    df["rsi_14"] = compute_rsi(df[close_col], 14)
    df['Target'] = df[close_col].shift(-1)
    
    print("Quantidade de valores nulos após criação de features:")
    print(df.isna().sum().sum())
    df = df.dropna().reset_index(drop=True)
    print('-'*64)
    print("Quantidade de linhas após remoção dos nulos:")
    print(len(df))
    print('-'*64)

    return df
df=create_features(df,'Data','Último','Vol.',windows=(5,10,20,50,75,100,150,200))
  ```
  
* Seleção de features:
  - Pipeline para seleção e validaçao temporal,  Avaliamos com ` GridSearchCV ` para escolher quantas e quais variáveis manter e para isso foi considerado:
    1- Modelo de  Ridge (escolhido por lidar bem com colunas altamente correlacionadas)
    2-  `TimeSeriesSplit` respeita ordem temporal:
    3- `GridSearchCV` para fazer a validação cruzada com os parametros determinados
    
 ```python
def select_features(train,test,target1=['Target','Último'],target='Target',k=10):
    """Seleciona as melhores features com base em um modelo de regressão.
    Parâmetros:
    train: DataFrame de treinamento.
    test: DataFrame de teste.
    target1: Lista de colunas alvo.
    target: Nome da coluna alvo principal.
    k: Número de features a serem selecionadas.
    Retorna:
    Lista de features selecionadas.  
    """
  


    
    #Seleção das colunas de features, excluindo colunas alvo e de data
    feature_cols=[col for col in train.columns if col not in target1 and col != 'Data' ]
    train = train[feature_cols + target1].dropna()
    test = test[feature_cols].dropna()
    X_train = train[feature_cols]
    y_train = train[target]
    
    


    def build_regression_selector(feature_cols, k=k):
        """Função para construir um pipeline de seleção de features e modelo de regressão.
        Parâmetros:
        feature_cols: Lista de colunas de features.
        k: Número de features a serem selecionadas.
        Retorna:
        Pipeline com seleção de features e modelo de regressão."""

        return Pipeline(steps=[
        ("scaler", StandardScaler()),
        #Algoritmo de seleção de features, usamos o KBest com f_regression pois é adequado para regressão onde a variável alvo é contínua e possui forte correlação linear com as features
        ("selector", SelectKBest(score_func=f_regression, k=k)),
        #Definição do modelo de regressão, usamos ridge pois é robusto e lida bem com a correlação entre features
        ("model", Ridge())
    ])

    # Configuração do TimeSeriesSplit para validação cruzada em séries temporais, definimos 5 splits pois é um valor comum que equilibra bem entre viés e variância
    tscv = TimeSeriesSplit(n_splits=5)
    # Construção do pipeline
    pipe = build_regression_selector(feature_cols, k=k)

    # Definição da grade de hiperparâmetros para busca em grid, otimizando o número de features e o parâmetro alpha do ridge
    # Aqui ajustamos os parametros para explorar melhor o modelo e os valores são selecionados com base em experimentação prévia
    param_grid = {
        "selector__k": [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
        "model__alpha": [0.1, 1.0, 5.0, 10.0]
}
    
    # Aqui chamamos o GridSearchCV para encontrar a melhor combinação de hiperparâmetros com base na métrica de erro absoluto médio negativo
    g = GridSearchCV(pipe, param_grid, cv=tscv, scoring="neg_mean_absolute_error")
    g.fit(X_train, y_train)


    # Extração das features selecionadas
    mask = g.best_estimator_.named_steps["selector"].get_support()
    #Passando para lista os nomes das features selecionadas
    selected_features = list(X_train.columns[mask])

   
    print("Selected Features:", selected_features)
    
    return selected_features
 ```

 * Decomposição sazonal

 ```python
result= seasonal_decompose(df['Último'], model='additive', period=30)
result.plot()
```
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/39cdbabc-fe80-4536-a59b-29ee323ebd64" />

 * Identificando ponto de interceptação com ACF
  ```python
# Função para plotar ACF
def plot_acf(data, lags=365, title="Autocorrelação da Série Temporal"):
    """Plota a função de autocorrelação (ACF) de uma série temporal.
        Parâmetros:
        data: Série temporal (array-like).
        lags: Número de lags a serem exibidos.
        title: Título do gráfico.
        """

    fig, ax = plt.subplots(figsize=(15, 5))
    #Chamando a função de plotagem da ACF do statsmodels
    _plot_acf(data, lags=lags, ax=ax, alpha=0.1)
    ax.set_title(title)
    ax.set_xlabel("Lags")
    ax.set_ylabel("Autocorrelação")
    fig.set_tight_layout(True)
    plt.show()

    #Função para identificar o ponto de interceptação dos coeficientes ACF com o intervalo de confiança
    acf_x = sm.tsa.acf(data, alpha=0.1, nlags=365)
    acf, ci = acf_x[:2]
    # Identificando o ponto de interceptação
    intercept_point = np.where(abs(acf) < (ci[:, 1] - acf))
    if intercept_point[0].size > 0:
        print(
            f"Coeficiente de interceptação da ACF no lag {intercept_point[0][0]}"
        )
    else:
        print("Os coeficientes da ACF não interceptam o limite")

plot_acf(df["Último"], lags=180)
```
<img width="1489" height="490" alt="image" src="https://github.com/user-attachments/assets/e86ca896-ada1-426b-9e8d-31154ca606ba" />
`Coeficiente de interceptação da ACF no lag 70`

 * Plotando o PACF
    ```python
    # Função para plotar PACF
    def plot_pacf(data, lags=30, title="Autocorrelação Parcial da Série Temporal"):
    """Plota a função de autocorrelação parcial (PACF) de uma série temporal.
        Parâmetros:
        data: Série temporal (array-like).
        lags: Número de lags a serem exibidos.
        title: Título do gráfico.
        """

     fig, ax = plt.subplots(figsize=(15, 5))
     #Chamando a função de plotagem da PACF do statsmodels
     _plot_pacf(data, lags=lags, ax=ax, alpha=0.05)
     ax.set_title(title)
     ax.set_xlabel("Lags")
     ax.set_ylabel("Autocorrelação Parcial")
     fig.set_tight_layout(True)
     plt.show()
     pacf_x = sm.tsa.pacf(data, alpha=0.05, nlags=lags)
     pacf, ci = pacf_x[:2]
     intercept_point = np.where(abs(pacf) < (ci[:, 1] - pacf))
     if intercept_point[0].size > 0:
         print(
             f"Coeficiente de interceptação da PACF no lag {intercept_point[0][0]}"
         )
     else:
         print("Os coeficientes da PACF não interceptam o limite")
    
   plot_pacf(df["Último"])
    ```
    <img width="1490" height="490" alt="image" src="https://github.com/user-attachments/assets/efc6fc7f-129b-4446-bf46-ec355240d55a" />
    
 ` Coeficiente de interceptação da PACF no lag 2`
 * Verificando a estacionaridade dos dados
    ```python
     # Teste de Dickey-Fuller aumentado (ADF)
     # Aqui verificamos a estacionariedade da série temporal usando o teste ADF
     adf_result = sm.tsa.adfuller(df["Último"])
     print(f"ADF Statistic: {adf_result[0]}")
     print(f"p-value: {adf_result[1]}")
     print(f"Critical Values: {adf_result[4]}")
     
     # Interpretamos uma serie menor que 0.05 como estacionária
     if adf_result[1] < 0.05:
         print("A série temporal é estacionária")
     else:
         print("A série temporal não é estacionária")
    ```
- Saída:
```python
        `ADF Statistic: -1.0021031938977978`
        `p-value: 0.7524964368015556`
        `Critical Values: {'1%': -3.439006442437876, '5%': -2.865360521688131, '10%': -2.5688044403756587}`
        `A série temporal não é estacionária`
```

* Aplicando a Diferenciação

  ```python
  # Teste de Dickey-Fuller aumentado (ADF)
  # Aplicando a diferenciação
  adf_result = sm.tsa.adfuller(df["Último"].diff().dropna())
  print(f"ADF Statistic: {adf_result[0]}")
  print(f"p-value: {adf_result[1]}")
  print(f"Critical Values: {adf_result[4]}")
  
  if adf_result[1] < 0.05:
      print("A série temporal é estacionária")
  else:
      print("A série temporal não é estacionária")
  ```
- Saída:
 ```python
ADF Statistic: -27.881563385332615
p-value: 0.0
Critical Values: {'1%': -3.4390179167598367, '5%': -2.8653655786032237, '10%': -2.5688071343462777}
A série temporal é estacionária
```
* Definindo p,q,d :

 ```python
# Determinação dos parâmetros p, d, q para o modelo ARIMA
#função sm.tsa.acf de autocorrelação para trazer o valor do lag onde a ACF cruza o intervalo de confiança representado por q
# identificação do q pelo ponto de interceptação ao 90% com o acf
acf_x = sm.tsa.acf(df["Último"], alpha=0.1, nlags=365)

acf, ci = acf_x[:2]

q = np.where(abs(acf) < (ci[:, 1] - acf))[0][0]

pacf_x = sm.tsa.pacf(df["Último"], alpha=0.1, nlags=30)
pacf, ci = pacf_x[:2]

p = np.where(abs(pacf) < (ci[:, 1] - pacf))[0][0]

d = 1  # apenas 1 diferenciação foi necessária para tornar a série estacionária
```

` p,q,d = (2, 70, 1)`


* Definindo a ordem sazonal
  ```python
   # Aqui usamos a função pm auto_arima para determinar os melhores parâmetros sazonais do modelo ARIMA
   # passamos m=5 pois as negociações na bolsa ocorrem em dias úteis, resultando em um padrão semanal de 5 dias
   # Também usamos stepwise=True para acelerar o processo de busca dos melhores parâmetros
   # Usamos a série original sem diferenciação, pois o auto_arima já lida com isso internamente 
   model = pm.auto_arima(
       df["Último"],
       seasonal=True,
       m=5,
       stepwise=True,
   )
   
   # Resumo do modelo ARIMA ajustado
   print(model.summary())
   
   # Detected seasonal order
   seasonal_order = model.seasonal_order
   print(f"Detected Seasonal Order: {seasonal_order}")
  ```
  - Saída:
     ```python
     SARIMAX Results                                
       ==============================================================================
       Dep. Variable:                      y   No. Observations:                  759
       Model:               SARIMAX(2, 1, 2)   Log Likelihood               -1213.291
       Date:                Mon, 27 Oct 2025   AIC                           2436.581
       Time:                        21:47:15   BIC                           2459.735
       Sample:                             0   HQIC                          2445.498
                                       - 759                                         
       Covariance Type:                  opg                                         
       ==============================================================================
                        coef    std err          z      P>|z|      [0.025      0.975]
       ------------------------------------------------------------------------------
       ar.L1         -0.3442      0.018    -18.814      0.000      -0.380      -0.308
       ar.L2         -0.9764      0.017    -57.858      0.000      -1.009      -0.943
       ma.L1          0.3517      0.028     12.788      0.000       0.298       0.406
       ma.L2          0.9450      0.027     34.957      0.000       0.892       0.998
       sigma2         1.4374      0.068     21.021      0.000       1.303       1.571
       ===================================================================================
       Ljung-Box (L1) (Q):                   0.29   Jarque-Bera (JB):                 5.66
       Prob(Q):                              0.59   Prob(JB):                         0.06
       Heteroskedasticity (H):               0.89   Skew:                            -0.04
       Prob(H) (two-sided):                  0.36   Kurtosis:                         3.41
       ===================================================================================
       
       Warnings:
       [1] Covariance matrix calculated using the outer product of gradients (complex-step).
       Detected Seasonal Order: (0, 0, 0, 5)
     ```

  ### Separação do DataSet

  ```python
  #Aqui realizamos a separação do dataset de acordo com parametros do tech challenge
  train_size = df.shape[0] - 30
  train,test= df[:train_size], df[-30:]
  ```

  * Chamada para a função de selecionar as features:

    ```python
    selected_features=select_features(train,test)
    ```
    - Saída:
       ` Selected Features: ['Abertura', 'Máxima', 'Mínima', 'mm_5', 'ema_5', 'mm_10', 'ema_10', 'mm_20', 'ema_20'] `

---


* Modelo Sarimax= ARIMA com:
    * componente autorregressivo (p),
    * diferença (d),
    * média móvel (q),
    * sazonalidade (P, D, Q, m),
    * mais **variáveis exógenas** (as features selecionadas).
 
  Escolhas importantes:

   * `d` e `D` definidos com base em testes de estacionariedade (ex: ADF).
   * `p` e `q` inferidos a partir de ACF/PACF.
   * Sazonalidade estimada com apoio de `auto_arima` para encontrar um padrão repetitivo.
   * Exógenas = sinais derivados de tendência, considerando variáveis selecionadas pelo SelectKBest.
   
   Por que SARIMAX?
   
   * Ele modela o “comportamento interno” do preço (inércia histórica) e também deixa o modelo ouvir sinais externos tipo: “o preço está muito acima da média de 5 dias?”, “o volume estourou hoje?”.
   * Isso normalmente melhora previsão de curto prazo (1 passo à frente).
   
   ---

  * Passando as variaveis selecionadas com exógenas para o modelo
    ```python
    exog_train = train[selected_features]
    exog_test= test[selected_features]

    ```

   ``` 
  # Modelo Sarimax
  # Aqui usamos a classe SARIMAX da biblioteca statsmodels para ajustar o modelo SARIMAX com os parâmetros p, d, q e a ordem sazonal determinada anteriormente
  # Passamos todos os dados de treino com as variaveis selecionadas como exógenas
  model = sm.tsa.statespace.SARIMAX(
    
    train["Último"],
    exog=exog_train,
    order=(
        p,
        d,
        q,
    ), 
    seasonal_order=seasonal_order,
   )
  # passando os dados para ajustar o modelo
  results = model.fit()
  
  # previsão dos dados de teste
  # usamos a função get_forecast para obter as previsões para o período de teste, passando as variáveis exógenas correspondentes
  # o uso do predicted_mean nos dá os valores previstos diretamente
  
  sarimax = results.get_forecast(steps=len(test), exog=exog_test).predicted_mean
  ```

 - Chamando a função de métricas:
 
```python
     metrics_sarimax = calculate_metrics(test["Último"], sarimax)
     print("SARIMAX Metrics:")
     print_metrics(metrics_sarimax)
```
- Saída:
  ```python
   SARIMAX Metrics:
   MAE: 0.739099941500741
   MSE: 0.8189094658104054
   MAPE: 0.51 %
  ```
---
### Prophet ( com regressores )

Foi treinado Prophet com:
 * Sazonalidade,
 * Features como regressoras selecionadas previamente


Por que Prophet?

* Prophet lida bem com tendências estruturais e sazonalidades “de calendário”.
* É rápido de ajustar, produz previsões “calendário-aware”
* Aceita regressoras externas que carregam a informação técnica que o preço sozinho não explica.


```python
 model = Prophet( daily_seasonality=True,weekly_seasonality=True, yearly_seasonality=True)
for c in selected_features:
    model.add_regressor(c)

model.fit(train_prophet)


future = model.make_future_dataframe(periods=len(test))
future[selected_features] = pd.concat([train_prophet[selected_features], test_prophet[selected_features]], ignore_index=True)
forecast = model.predict(future)
preds_prophet = forecast['yhat'].iloc[-len(test):].values


metrics_prophet = calculate_metrics(test["Último"], preds_prophet)
print("Prophet Metrics:")
print_metrics(metrics_prophet)

```
- Saída:
  ```python
  Prophet Metrics:
  MAE: 0.44661990494293113
  MSE: 0.48100388416152257
  MAPE: 0.31 %

  ```


Também foi feita uma busca de hiperparâmetros via `cross_validation` + `performance_metrics`, variando:

* `changepoint_prior_scale`
* `seasonality_prior_scale`
* `seasonality_mode`
* `changepoint_range`

   ```python
   # Hiperparâmetro tuning para o modelo Prophet
   # o padrão do prophet é 0.05 para changepoint_prior_scale e 10.0 para seasonality_prior_scale
   # portanto aqui testamos valores mais baixos para changepoint_prior_scale e valores mais altos para seasonality_prior_scale
   # pois entendemos que o ibovespa tende a seguir tendencias mais suaves e tem forte sazonalidade anual e diária
   
   param_grid = {
       'changepoint_prior_scale': [0.01, 0.5, 0.1, 0.3],#Controla a flexibilidade das mudanças de tendência no modelo
       'seasonality_prior_scale': [1.0, 2.0, 3.0, 5.0],#Controla a força da sazonalidade no modelo
       'seasonality_mode': ['additive', 'multiplicative'], # Define o modo de sazonalidade 
       # Método Additive assume que a sazonalidade é constante ao longo do tempo, enquanto o Multiplicative assume que a sazonalidade varia proporcionalmente ao nível da série temporal
       'changepoint_range': [0.8, 0.7, 0.9, 1] #Define a proporção do histórico de dados em que os pontos de mudança podem ocorrer, o padrão é 0.8
   }
   
   all_params = [
       dict(zip(param_grid.keys(), v)) # Cria um dicionário de parâmetros para cada combinação
       for v in itertools.product(*param_grid.values()) # Gera todas as combinações possíveis dos valores dos hiperparâmetros
   ]
   
   
   metrics = []
   # Loop sobre todas as combinações de hiperparâmetros
   for params in all_params:
       # para cada combinação, criamos e ajustamos um modelo Prophet com os parâmetros atuais
   
       m = Prophet(**params, daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=False)
       # adicionando as variaveis exógenas
       for c in selected_features:
           m.add_regressor(c)
   
       m.fit(train_prophet)
       # Realizamos a validação cruzada temporal usando a função cross_validation do Prophet
       # Passamos o modelo ajustado, o periodo inicial de treinamento, o período entre as previsões e o horizonte de previsão
       # O período inicial de 540 dias corresponde de acordo com a quantidade de dados disponíveis
       # o período de 180 dias corresponde a um semestre de negociações
       # o horizonte de 180 dias corresponde a previsão para o próximo semestre
       
       df_cv = cross_validation(m, initial='540 days', period='180 days', horizon='180 days')
       # Calculamos as métricas de desempenho usando a função performance_metrics do Prophet
       df_p = performance_metrics(df_cv, rolling_window=0.1)
       metrics.append((params, df_p['rmse'].values[0]))
   print("Hyperparameter tuning completed.")
   print(metrics)
   best_params, best_rmse = min(metrics, key=lambda x: x[1])
   print("Best Parameters:", best_params)
   print("Best MSE:", best_rmse)
   
   
   m = Prophet(**best_params, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
   for c in selected_features:
       m.add_regressor(c)
   
   m.fit(train_prophet)
   
   
   future = m.make_future_dataframe(periods=len(test))
   future[selected_features] = pd.concat([train_prophet[selected_features], test_prophet[selected_features]], ignore_index=True)
   forecast = m.predict(future)
   preds_prophet_hype = forecast['yhat'].iloc[-len(test):].values
   
   
   metrics_prophet = calculate_metrics(test["Último"], preds_prophet_hype)
   print("Prophet Metrics:")
   print_metrics(metrics_prophet)

Saída:
```
   Prophet Metrics:
   MAE: 0.43840158265241297
   MSE: 0.45826273098011794
   MAPE: 0.30 %
```
---

### Plot com todos modelos mais dados de teste:

<img width="1160" height="624" alt="image" src="https://github.com/user-attachments/assets/d8e831a0-50ee-4ca7-9edc-c4f99ff929ba" />


### 8.1. Métricas de valor numérico (regressão)

Avaliado no conjunto de teste (últimos ~30 dias):

* **MAE (Mean Absolute Error):** erro médio em pontos de índice.
* **MSE / RMSE:** penaliza erros grandes com mais força.
* **MAPE (%):** erro percentual relativo.


Depois de prevermos o preço de amanhã (`y_pred`), transformamos isso em alta/queda:

```python
#Transformando em predicts em variaveis categoricas para calcular acurácia

test['real_up'] = ( test["Último"]-test['Último'].shift(1)) > 0

test['sarimax_up'] = (test['sarimax'] - test['sarimax'].shift(1) ) > 0

test['prophet_up'] = (test['prophet'] - test['prophet'].shift(1)) > 0

test['prophet2_up'] = (test['prophet2'] - test['prophet2'].shift(1)) > 0
```
---
Com isso calculamos:

* **Accuracy:** % de dias em que o modelo acertou se ia subir ou cair.
* **Balanced Accuracy:** média do acerto em “sobe” e “cai”, útil se a série está desequilibrada (por exemplo, mais dias de alta do que de queda).
---
Por fim:

* Calculamos as métricas que definem o sucesso do modelo de acordo com o exercício proposto:

  Modelo Sarimax:
  ```python
  cm_sarimax = confusion_matrix(real_up[1:], sarimax_up[1:])
  cm_sarimax_display = ConfusionMatrixDisplay(cm_sarimax, display_labels=["Baixa", "Alta"])
  print(f"Acurácia do modelo Sarimax: {accuracy_score(real_up[1:], sarimax_up[1:])}")
  print(f"Acurácia Balanceada do modelo Sarimax: {balanced_accuracy_score(real_up[1:], sarimax_up[1:])}")
  cm_sarimax_display.plot(cmap="Blues");

  ```

Acurácia do modelo Sarimax: 0.9310344827586207
Acurácia Balanceada do modelo Sarimax: 0.9375

Matriz de confusão Sarimax:
<img width="528" height="438" alt="image" src="https://github.com/user-attachments/assets/9da6d29c-949c-4643-9e0e-1bcd72032746" />

  Modelo Prophet sem hyperparametros:
```python
cm_prophet = confusion_matrix(real_up[1:], prophet_up[1:])
cm_prophet_display = ConfusionMatrixDisplay(cm_prophet, display_labels=["Baixa", "Alta"])
print(f"Acurácia do modelo Prophet Simples: {accuracy_score(real_up[1:], prophet_up[1:])}")
print(f"Acurácia Balanceada do modelo Prophet Simples: {balanced_accuracy_score(real_up[1:], prophet_up[1:])}")
cm_prophet_display.plot(cmap="Blues");

```
Acurácia do modelo Prophet Simples: 0.8620689655172413
Acurácia Balanceada do modelo Prophet Simples: 0.8677884615384616
<img width="528" height="432" alt="image" src="https://github.com/user-attachments/assets/4f903957-e8a3-4ea4-a8df-16d8ced2cbd8" />

  Modelo Prophet:
```python
cm_prophet = confusion_matrix(real_up[1:], prophet_up[1:])
cm_prophet_display = ConfusionMatrixDisplay(cm_prophet, display_labels=["Baixa", "Alta"])
print(f"Acurácia do modelo Prophet Simples: {accuracy_score(real_up[1:], prophet_up[1:])}")
print(f"Acurácia Balanceada do modelo Prophet Simples: {balanced_accuracy_score(real_up[1:], prophet_up[1:])}")
cm_prophet_display.plot(cmap="Blues");

```
Acurácia do modelo Prophet Simples: 0.8620689655172413
Acurácia Balanceada do modelo Prophet Simples: 0.8677884615384616
<img width="528" height="432" alt="image" src="https://github.com/user-attachments/assets/4f903957-e8a3-4ea4-a8df-16d8ced2cbd8" />

---
### Confiabilidade

O modelo é considerado aceitável se:

* Mantém erro de valor (MAE / MAPE) dentro de um patamar razoável (sem explosões em dias de stress).
* Consegue atingir ou se aproximar da meta de acurácia direcional ≥ 75% na janela de teste.
* Não depende de vazamento de futuro: todas as features são conhecidas até D e a previsão é feita apenas para D+1.

---

### Conclusão

- Nosso pipeline conseguiu uma boa performance com o modelo sarimax, definimos ele como nosso melhor modelo até então.
- A seleção de variáveis nos ajudou a reduzir os ruídos
- Conseguimos boas métricas em ambos os modelos

  
     

