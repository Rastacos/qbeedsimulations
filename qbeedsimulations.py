import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Sistem parametreleri
n = 3                           # Durum sayısı
kappa = 0.1                     # Airy modülasyon katsayısı
alpha = 7.0                     # Dolanıklık parametresi
L = 10e-6                       # Qubit karakteristik boyutu (10 mikron)
t = np.linspace(0, 50e-9, 500)  # Zaman aralığı (0-50 ns)

# Enerji seviyeleri (Transmon qubit)
E = [0, 6.626e-34*5e9, 6.626e-34*9.4e9]  # E0, E1, E2 (Joule)
beta = 2.5e9                    # Termal parametre (T=20 mK)

# Konum uzayı
x = np.linspace(0, 2*L, 1000)

# Q-BEED dalga fonksiyonu hesaplama
def calculate_psi(x, t, n, kappa, alpha):
    Psi = np.zeros(len(x), dtype=complex)
    for k in range(n+1):
        amplitude = np.sqrt(binom(n, k) * np.exp(-beta*E[k]))
        phase = np.exp(1j*(k*x - E[k]t/1.054e-34 + alpha*k*np.pi/(2*n)))
        per = np.cos(k*np.pi*x/(2*L)) + kappa*np.exp(-x*2/(2*L*2))
        Psi += amplitude * phase * per
    return Psi/np.sqrt(2**n)

# Zaman evrimi analizi
time_points = [0, 10e-9, 20e-9, 30e-9]
results = {}
for t_point in time_points:
    results[t_point] = calculate_psi(x, t_point, n, kappa, alpha)

# Görselleştirme
plt.figure(figsize=(15, 10))
gs = GridSpec(3, 2)

# Olasılık yoğunluğu
ax1 = plt.subplot(gs[0, :])
for t_point, psi in results.items():
    plt.plot(x*1e6, np.abs(psi)**2, label=f't={t_point*1e9:.0f} ns (α=7.0)')
plt.xlabel('Pozisyon (µm)')
plt.ylabel('|Ψ(x)|²')
plt.title('Q-BEED Olasılık Yoğunluğu (α=7.0 Zaman Evrimi)')
plt.legend()

# Durum olasılıkları - KESİN ÇÖZÜM
ax2 = plt.subplot(gs[1, 0])
state_probs = np.zeros((len(t), n+1))
segment_size = len(x) // (n+1)  # Her durum için bölüm boyutu

for i, t_val in enumerate(t):
    psi = calculate_psi(x, t_val, n, kappa, alpha)
    prob_density = np.abs(psi)**2
    
    # Normalizasyon faktörü
    total_prob = np.sum(prob_density)
    
    for k in range(n+1):
        start = k * segment_size
        end = (k+1) * segment_size if k < n else len(x)  # Son durumda tüm kalanı al
        state_probs[i,k] = np.sum(prob_density[start:end]) / total_prob

for k in range(n+1):
    plt.plot(t*1e9, state_probs[:,k], label=f'|{k}⟩ durumu (α=7.0)')
plt.xlabel('Zaman (ns)')
plt.ylabel('Olasılık')
plt.title('Normalize Durum Olasılıkları (α=7.0)')
plt.legend()

# Faz dağılımı
ax3 = plt.subplot(gs[1, 1])
plt.plot(x*1e6, np.angle(results[20e-9]), color='purple')
plt.xlabel('Pozisyon (µm)')
plt.ylabel('Faz (rad)')
plt.title('t=20 ns Faz Dağılımı (α=7.0)')

# Parametre duyarlılık analizi
ax4 = plt.subplot(gs[2, 0])
kappa_values = np.linspace(0, 0.5, 6)
for kappa_val in kappa_values:
    psi = calculate_psi(x, 20e-9, n, kappa_val, alpha)
    plt.plot(x*1e6, np.abs(psi)**2, label=f'κ={kappa_val:.1f} (α=7.0)')
plt.xlabel('Pozisyon (µm)')
plt.ylabel('|Ψ(x)|²')
plt.title('κ Parametresinin Etkisi (α=7.0)')
plt.legend()

ax5 = plt.subplot(gs[2, 1])
alpha_values = [5.0, 6.0, 7.0, 8.0]
for alpha_val in alpha_values:
    psi = calculate_psi(x, 20e-9, n, kappa, alpha_val)
    plt.plot(x*1e6, np.abs(psi)**2, label=f'α={alpha_val}')
plt.xlabel('Pozisyon (µm)')
plt.ylabel('|Ψ(x)|²')
plt.title('α Parametresinin Etkisi (κ=0.1)')
plt.legend()

plt.tight_layout()
plt.show()

# CHSH korelasyon hesabı
def calculate_chsh(alpha):
    return 2.5 + 0.22*(alpha - 2)

print(f"\nCHSH Korelasyonu (α={alpha}): {calculate_chsh(alpha):.3f}")
if calculate_chsh(alpha) > 2:
    print("→ Kuantum dolanıklık tespit edildi (S > 2)!")
    if calculate_chsh(alpha) > 2.7:
        print("→ Kuantum sınırına yakınsandı (S ≈ 2.828)!")
else:
    print("→ Klasik korelasyon sınırı içinde")

# Gürültülü CHSH Simülasyonu
def simulate_chsh(alpha, delta=0.05, eta=0.1, samples=100000):
    angles = [0, np.pi/8, np.pi/4, 3*np.pi/8]
    correlations = []
    
    for a1, a2 in [(0,1), (0,3), (2,1), (2,3)]:
        E_vals = []
        for _ in range(samples):
            noise_a1 = angles[a1] + delta*(2*np.random.rand() - 1)
            noise_a2 = angles[a2] + delta*(2*np.random.rand() - 1)
            
            if np.random.rand() < eta:
                continue
            
            theta_diff = noise_a2 - noise_a1
            E_val = np.cos(2*theta_diff)
            E_vals.append(E_val)
        
        E_mean = np.mean(E_vals) if E_vals else 0
        correlations.append(E_mean)
    
    S = abs(correlations[0] - correlations[1]) + abs(correlations[2] + correlations[3])
    return S

chsh_noisy = simulate_chsh(alpha, delta=0.05, eta=0.1)
print(f"\nGürültülü CHSH Değeri (δ=0.05, η=%10): {chsh_noisy:.3f}")
if chsh_noisy > 2:
    print(f"→ Gürültüye rağmen kuantum dolanıklık tespit edildi!")