def calcular_score(row):
    score = 0
    score += (8 - row["sueno_horas"]) * 0.5
    score += row["cafe_tazas"] * 0.4
    score -= row["ejercicio_min"] * 0.005
    score += row["pantallas_horas"] * 0.2
    score -= row["estudio_horas"] * 0.01
    score += row["comida_chatarra"] * 0.8
    return score

def calcular_estres(score, p25, p75):
    if score < p25:
        return 0  # Bajo
    elif score < p75:
        return 1  # Medio
    else:
        return 2  # Alto