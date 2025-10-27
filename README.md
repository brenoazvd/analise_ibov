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

* Criação de colunas de datas
    ```
  df['ano']=df['Data'].dt.year
  df['mes']=df['Data'].dt.month
  df['dow']=df['Data'].dt.day_of_week
  df['Anomes']=df['Data'].dt.strftime('%m.%y')

### Engenharia de atributos

* Criação de Features:
  ```
  #Função para criação de features
def create_features(df,date_col,close_col,vol_col,windows=(5,10,20,50,100)):
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
* 


*   ``
*   
* 

