# Previsão Direcional do IBOVESPA (D+1)

## 1.Objetivo:

Prever se  fechamento do IBOVESPA do **Próximo dia (d+1)** será **maior ↑**  ou **menor ↓** do que o fechamento de hoje.

** Meta do desafio:** Atingir pelo menos 75% de acurácia nos dados de teste, que devem ser os últimos 30 dias


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
  
    ```
        
        df['Data']=pd.to_datetime(df['Data'],format='%d.%m.%Y')
   ```
* Conversão de `Vol.` para numero:

     ```
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
  ```
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
    ```
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

* Criação de Features:
  ```
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
      # 1) ordem temporal
      df = df.sort_values(date_col).reset_index(drop=True)
 
     # 2) retornos básicos
     df['ret_1'] = df[close_col].pct_change(1)
     df["ret_5d"]  = df[close_col].pct_change(5)
     df["ret_10d"] = df[close_col].pct_change(10)
 
 
     # 3) médias e EMAs multijanelas
     for w in windows:
         df[f'mm_{w}']  = df[close_col].rolling(window=w, min_periods=w).mean()
         df[f'ema_{w}'] = df[close_col].ewm(span=w, adjust=False).mean()
         # spread preço - média
         df[f'spread_mm{w}'] = df[close_col] - df[f'mm_{w}']
         # MÉDIA de volume
         
         df[f'volm_{w}'] = df[vol_col].rolling(window=w, min_periods=w).mean()
 
     # 4) relações preço vs média (com eps para evitar div/0)
     eps = 1e-9
     for w in windows:
         df[f'close_over_mm{w}'] = (df[close_col] / (df[f'mm_{w}'] + eps)) - 1
 
     # 5) momentums multijanelas
     for k in (3,5,10):
         df[f'momentum_{k}'] = df[close_col].pct_change(k)
 
     # 6) Var% como numérico e shifts
     
     df['var_shift_1'] = df['Var%'].shift(1)
     df['var_shift_2'] = df['Var%'].shift(2)
     df['var_shift_3'] = df['Var%'].shift(3)
 
     # 7) razão de volume curto/longo 
     if all(f'volm_{w}' in df for w in (5,20)):
         base = df['volm_20'].replace(0, np.nan)
         df['vol_ratio_5_20'] = (df['volm_5'] / base)
 
 
 
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
 ```
 def select_features(train,test,target1=['Target','Último'],target='Target',k=10):
    feature_cols=[col for col in train.columns if col not in target1 and col != 'Data' ]
    train = train[feature_cols + target1].dropna()
    test = test[feature_cols].dropna()
    X_train = train[feature_cols]
    X_test = test[feature_cols]
    y_train = train[target]
    
    X_train.dropna(inplace=True)
    def build_regression_selector(feature_cols, k=10):
        return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_regression, k=k)),
        ("model", Ridge())
    ])

    
    tscv = TimeSeriesSplit(n_splits=5)
    pipe = build_regression_selector(feature_cols, k=10)

   
    param_grid = {
        "selector__k": [5, 10, 15, 20],
        "model__alpha": [0.1, 1.0, 5.0, 10.0]
}
    g = GridSearchCV(pipe, param_grid, cv=tscv, scoring="neg_mean_absolute_error")
    g.fit(X_train, y_train)

    mask = g.best_estimator_.named_steps["selector"].get_support()
    selected_features = list(X_train.columns[mask])

   
    print("Selected Features:", selected_features)
    return selected_features
 ```
<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/39cdbabc-fe80-4536-a59b-29ee323ebd64" />

<img width="1005" height="547" alt="image" src="https://github.com/user-attachments/assets/d6a904ac-22a0-4db7-b8e0-f6b0088add30" />


