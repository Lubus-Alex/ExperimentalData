# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import re

# ============================================================
# 1. ПАРСИНГ ФАЙЛА SpecsLab2
# ============================================================
def parse_specs_xps(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.split(r'# values in kinetic energy\n# Region:\s+', content)
    regions = []
    for block in blocks[1:]:
        lines = block.split('\n')
        name = lines[0].strip()
        hv, wf = 1253.6, 4.34
        for ln in lines[:30]:
            if ln.startswith('# Excitation Energy:'): hv = float(ln.split(':')[1])
            if ln.startswith('# Eff. Workfunction:'): wf = float(ln.split(':')[1])
        ke, cnt, tr = [], [], []
        in_data = False
        for ln in lines:
            if ln.startswith('# ColumnLabels:'):
                in_data = True; continue
            if in_data and ln.strip() and not ln.startswith('#'):
                p = ln.split()
                if len(p) >= 3:
                    ke.append(float(p[0])); cnt.append(float(p[1])); tr.append(float(p[2]))
        if ke:
            regions.append(dict(name=name, hv=hv, wf=wf, ke=np.array(ke),
                                counts=np.array(cnt), trans=np.array(tr)))
    return regions

# ============================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================
def moving_avg(y, w=5):
    out = np.convolve(y, np.ones(w)/w, mode='same')
    out[:w//2] = y[:w//2]; out[-(w//2):] = y[-(w//2):]
    return out

def noise_sigma(y):
    """Оценка шума по остатку от скользящего среднего."""
    return np.std(y - moving_avg(y, 5))

def linear_bg(y):
    return np.linspace(y[0], y[-1], len(y))

def shirley_bg(be, y, n_iter=200, tol=1e-6):
    """Итеративный фон Ширли (массив по BE по возрастанию).
    Если фон убывает (Shirley неприменим) — автоматический откат на линейный."""
    if y[-1] <= y[0]:
        return linear_bg(y)
    b = np.full_like(y, y[0])
    for _ in range(n_iter):
        net = np.clip(y - b, 0.0, None)
        cum = np.zeros_like(y)
        cum[1:] = np.cumsum((net[1:] + net[:-1]) * 0.5 * np.diff(be))
        if cum[-1] <= 0:
            return linear_bg(y)
        b_new = y[0] + (y[-1] - y[0]) * cum / cum[-1]
        if np.max(np.abs(b_new - b)) < tol * max(1.0, y.max()):
            return b_new
        b = b_new
    return b

# ============================================================
# 3. НАСТРОЙКИ (шкала ПОСЛЕ калибровки по C 1s!)
#    search — окно поиска пика, off — границы интегрирования
#    относительно найденного максимума (учитывает дублеты)
# ============================================================
CALIB_C1S = 284.8          # None — отключить калибровку
BG_METHOD = 'shirley'      # 'shirley' | 'linear'

CONFIG = {
    'O':  dict(region='O 1s',       search=(527, 536),  off=(-4.0, +4.0),  rsf=2.93, in_total=True),
    'N':  dict(region='N 1s Nb 3p', search=(395, 401),  off=(-3.0, +3.0),  rsf=1.80, in_total=True),
    'Nb': dict(region='Nb 3d',      search=(203, 212),  off=(-3.2, +6.0),  rsf=3.00, in_total=True),
    'Na': dict(region='Na 1s',      search=(1069,1077), off=(-3.5, +3.5),  rsf=2.08, in_total=True),
    'Ga': dict(region='Ga 2p',      search=(1112,1125), off=(-3.0, +31.0), rsf=5.96, in_total=True),
    'C':  dict(region='C 1s',       search=(282, 290),  off=(-3.5, +3.5),  rsf=1.00, in_total=False),
}

# ============================================================
# 4. РАСЧЕТ
# ============================================================
def analyze(regions):
    # --- перевод в BE и коррекция пропускания ---
    for r in regions:
        r['be'] = r['hv'] - r['ke'] - r['wf']
        r['y']  = r['counts'] / np.where(r['trans'] == 0, 1e-10, r['trans'])
        idx = np.argsort(r['be'])
        r['be'], r['y'] = r['be'][idx], r['y'][idx]

    get = lambda reg: next(r for r in regions if reg in r['name'])

    # --- калибровка по C 1s ---
    shift = 0.0
    if CALIB_C1S is not None:
        rc = get('C 1s'); c = CONFIG['C']
        m = (rc['be'] > c['search'][0]) & (rc['be'] < c['search'][1])
        shift = CALIB_C1S - rc['be'][m][np.argmax(moving_avg(rc['y'][m], 5))]
        print(f"Зарядовая калибровка по C 1s: сдвиг = {shift:+.2f} эВ")
        for r in regions:
            r['be'] = r['be'] + shift

    results = []
    for el, c in CONFIG.items():
        r = get(c['region'])
        # окно поиска пика -> центр
        m = (r['be'] > c['search'][0]) & (r['be'] < c['search'][1])
        i0 = np.where(m)[0][np.argmax(moving_avg(r['y'][m], 5))]
        center = r['be'][i0]
        # окно интегрирования вокруг центра
        w = (r['be'] >= center + c['off'][0]) & (r['be'] <= center + c['off'][1])
        be_w, y_w = r['be'][w], r['y'][w]
        # фон
        bg = shirley_bg(be_w, y_w) if BG_METHOD == 'shirley' else linear_bg(y_w)
        net = np.clip(y_w - bg, 0.0, None)      # <-- отрицательные значения обрезаются
        sigma = noise_sigma(y_w)
        below_dl = net.max() < 3 * sigma        # пик не обнаружен (предел обнаружения)
        area = 0.0 if below_dl else np.trapz(net, be_w)
        results.append(dict(el=el, center=center, area=area, rsf=c['rsf'],
                            in_total=c['in_total'], below_dl=below_dl,
                            be=be_w, y=y_w, bg=bg, net=net))
    return results

def report(results):
    total = sum(r['area']/r['rsf'] for r in results if r['in_total'] and r['area'] > 0)
    print("\n" + "="*78)
    print(f"{'Эл.':<5}{'Пик, эВ':>9}{'Площадь':>12}{'RSF':>7}{'at.%':>9}   Примечание")
    print("-"*78)
    for r in results:
        if r['below_dl']:
            conc, note = 0.0, "< предела обнаружения (0.00)"
        else:
            s = r['area']/r['rsf']
            conc = (s/total*100) if (r['in_total'] and total > 0) else \
                   (s/(total+s)*100 if not r['in_total'] else 0.0)
            note = "" if r['in_total'] else "(адвентивный, вне баланса)"
        print(f"{r['el']:<5}{r['center']:>9.2f}{r['area']:>12.1f}{r['rsf']:>7.2f}{conc:>9.2f}   {note}")
    print("="*78)

def plot_results(results):
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 4*len(results)),
                             constrained_layout=True)
    for ax, r in zip(axes, results):
        ax.plot(r['be'], r['y'], 'b-', lw=1.2, label='Спектр')
        ax.plot(r['be'], r['bg'], 'r--', lw=1.2, label=f'Фон ({BG_METHOD})')
        ax.fill_between(r['be'], r['bg'], r['bg'] + r['net'], color='green',
                        alpha=0.3, label='Площадь (≥0)')
        ax.set_xlim(r['be'].max(), r['be'].min())
        ax.set_ylabel('Counts/s / Trans')
        ax.set_title(f"{r['el']} — пик {r['center']:.2f} эВ"
                     + ("  [ниже предела обнаружения]" if r['below_dl'] else ""),
                     fontweight='bold')
        ax.legend(loc='upper right'); ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Binding Energy (eV)')
    plt.savefig('XPS_quant_fixed.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    regions = parse_specs_xps(r'D:\Users\Lub\ExperimentalData\Data\NbN(5)GaN  t0 -Mg -EP20.xy')
    res = analyze(regions)
    report(res)
    plot_results(res)