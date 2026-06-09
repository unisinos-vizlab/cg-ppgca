# Monitor de Atenção do Motorista — OpenPilot DM

Aplicativo de visão computacional para verificar a atenção do motorista em tempo real, baseado no sistema **Driver Monitoring** do [openpilot](https://github.com/commaai/openpilot) (comma.ai).

## Arquitetura

```
┌─────────────────────────────────────────────────────────┐
│                    Pipeline Principal                    │
│                                                         │
│  Câmera ──► FaceAnalyzer ──► DriverAttentionMonitor     │
│               (MediaPipe       (Política do openpilot)  │
│              + solvePnP)               │                │
│                                        ▼                │
│                               AttentionDisplay          │
│                               (Dashboard OpenCV)        │
└─────────────────────────────────────────────────────────┘
```

### Componentes

| Módulo | Descrição |
|--------|-----------|
| `face_analyzer.py` | Detecção de rosto com **MediaPipe FaceMesh** + estimativa de pose de cabeça via **cv2.solvePnP** + detecção de piscada com **Eye Aspect Ratio (EAR)** |
| `monitor.py` | Política de atenção portada diretamente do `selfdrive/monitoring/policy.py` do openpilot. Mantém um contador de *awareness* (0–1) que decresce quando distraído e aumenta quando atento |
| `display.py` | Dashboard em tempo real: vídeo com sobreposições, gauge circular de atenção, indicadores de distração, barra de status |
| `main.py` | Loop principal, controles de teclado, gravação opcional |

## Lógica de Atenção (baseada no openpilot)

O sistema detecta **3 tipos de distração**:

| Tipo | Condição |
|------|----------|
| **Pose** | Desvio de yaw > 23° **ou** pitch para cima > 18° em relação à posição calibrada |
| **Olhos** | Probabilidade média de piscada > 0.865 (olhos fechados por tempo excessivo) |
| **Celular** | Probabilidade de uso de celular > 50% |

**Níveis de alerta** (padrão Euro NCAP / openpilot):

| Nível | Timeout | Cor |
|-------|---------|-----|
| 0 — Normal | — | Verde |
| 1 — Aviso | 3 s | Amarelo |
| 2 — Perigo | 5 s | Laranja |
| 3 — Terminal | 11 s | Vermelho |

A variável `awareness` decresce a cada frame enquanto distraído e se recupera quando atento, usando um filtro de primeira ordem (τ = 0.25 s) para suavizar oscilações.

## Instalação

```bash
# Python 3.9+
pip install -r requirements.txt
```

### Dependências
- `opencv-python >= 4.8`
- `mediapipe >= 0.10`
- `numpy >= 1.24`

## Uso

```bash
# Webcam padrão (índice 0)
python -m driver_attention.main

# Câmera alternativa
python -m driver_attention.main --source 1

# Arquivo de vídeo
python -m driver_attention.main --source video.mp4

# Salvar saída
python -m driver_attention.main --save saida.mp4
```

### Controles de teclado

| Tecla | Ação |
|-------|------|
| `q` / `ESC` | Sair |
| `r` | Reset completo (awareness + calibração) |
| `c` | Limpar calibração de pose |
| `SPACE` | Pausar / Retomar |

## Display

```
┌──────────────────────────┬─────────────────────┐
│  Vídeo com overlays      │  Gauge de atenção   │
│  • Malha facial          │  [●●●●●●○○○○] 60%   │
│  • Eixos de pose (RGB)   │                     │
│  • Indicadores de        │  ● Face detectada   │
│    piscada               │  ● Pose OK          │
│                          │  ○ Olhos fechados   │
│  Borda colorida por      │  ● Sem celular      │
│  nível de alerta         │                     │
│                          │  [ALERTA NÍVEL 2]   │
└──────────────────────────┴─────────────────────┘
│ Pitch: +2.1° | Yaw: -5.3° | Blink E: 0.12 ...  │
└─────────────────────────────────────────────────┘
```

## Referências

- [openpilot — selfdrive/monitoring/policy.py](https://github.com/commaai/openpilot/blob/master/selfdrive/monitoring/policy.py)
- [openpilot — selfdrive/modeld/dmonitoringmodeld.py](https://github.com/commaai/openpilot/blob/master/selfdrive/modeld/dmonitoringmodeld.py)
- Euro NCAP Driver Engagement Protocol v1.1
- Soukupová & Čech (2016) — Eye Aspect Ratio para detecção de sonolência
- MediaPipe FaceMesh — [mediapipe.dev](https://mediapipe.dev)
