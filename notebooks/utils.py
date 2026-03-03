import numpy as np

G = 1.0
G_ASTS = 100.0

Z0 = 0.0
ZF = np.log(50000.0)

W0 = np.log(45 / np.sqrt(32 * np.pi**7) * G / G_ASTS) - 1.0
WF = -26.1586

GAMMA_STD = 0.0
GAMMA_RS = 2.0
GAMMA_GB = -2.0/3.0

C_STD = 29.2037
C_RS = 31.7022
C_GB = 27.6359

ZT1 = np.log(500.0)
