import pandas as pd


def detectar_cambios_porcentuales(df: pd.DataFrame, variable: str, umbral: float) -> pd.DataFrame:
    """
    Detecta cambios porcentuales superiores al umbral en una variable específica
    respecto al valor anterior en orden temporal.
    """
    # Asegurar tipos correctos
    df = df.copy()
    df = df[df['VarName'] == variable].sort_values(by='Time_ms')


    # Convertir valores a numérico
    df['VarValue'] = pd.to_numeric(df['VarValue'].astype(str).str.replace(",", "."), errors='coerce')
    df = df.dropna(subset=['VarValue'])


    # Calcular variación porcentual respecto al valor anterior
    df['Cambio_%'] = df['VarValue'].pct_change() * 100


    # Filtrar cambios significativos
    df_cambios = df[abs(df['Cambio_%']) >= umbral].copy()


    return df_cambios
