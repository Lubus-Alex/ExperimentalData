import numpy as np
import matplotlib.pyplot as plt
import re

# ==========================================
# 1. ПАРСИНГ ДАННЫХ ИЗ ФАЙЛА SPECSLAB
# ==========================================
def parse_specs_xps(filename):
    """Читает файл SpecsLab2 и возвращает список регионов."""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Разделяем текст на блоки по маркеру начала нового региона
    blocks = re.split(r'# values in kinetic energy\n# Region:\s+', content)
    regions = []
    
    for block in blocks[1:]:
        lines = block.split('\n')
        name = lines[0].strip()
        
        hv, wf = 1253.6, 4.34 # Значения по умолчанию
        
        # Поиск метаданных
        for line in lines[:30]:
            if 'Excitation Energy:' in line:
                hv = float(line.split(':')[1].strip())
            if 'Eff. Workfunction:' in line:
                wf = float(line.split(':')[1].strip())
                
        # Поиск числовых данных
        in_data = False
        ke, counts, trans = [], [], []
        for line in lines:
            if '# ColumnLabels:' in line:
                in_data = True
                continue
            if in_data and line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        ke.append(float(parts[0]))
                        counts.append(float(parts[1]))
                        trans.append(float(parts[2]))
                    except ValueError:
                        pass
                        
        if ke:
            regions.append({
                'name': name, 'hv': hv, 'wf': wf,
                'ke': np.array(ke),
                'counts': np.array(counts),
                'trans': np.array(trans)
            })
    return regions

# ==========================================
# 2. НАСТРОЙКИ ЭЛЕМЕНТОВ И RSF
# ==========================================
# RSF (Scofield cross-sections для Mg Kα). 
# При необходимости скорректируйте значения RSF под ваши стандарты.
ELEMENT_CONFIG = {
    'O':  {'region': 'O 1s',       'be_min': 525.0, 'be_max': 536.0, 'rsf': 2.93, 'include_in_total': True},
    'N':  {'region': 'N 1s Nb 3p', 'be_min': 394.0, 'be_max': 402.0, 'rsf': 1.80, 'include_in_total': True},
    'Nb': {'region': 'Nb 3d',      'be_min': 198.0, 'be_max': 215.0, 'rsf': 3.00, 'include_in_total': True}, # ~3.0-3.5 для Specs
    'Ga': {'region': 'Ga 2p',      'be_min': 1110.0,'be_max': 1150.0,'rsf': 5.96, 'include_in_total': True},
    'Na': {'region': 'Na 1s',      'be_min': 1066.0,'be_max': 1078.0,'rsf': 2.08, 'include_in_total': True},
    'C':  {'region': 'C 1s',       'be_min': 281.0, 'be_max': 293.0, 'rsf': 1.00, 'include_in_total': False} # Исключаем из баланса
}

# ==========================================
# 3. РАСЧЕТ КОНЦЕНТРАЦИЙ
# ==========================================
def calculate_concentrations(parsed_regions, config):
    results = []
    
    for element, params in config.items():
        # Ищем соответствующий регион в файле
        region_data = next((r for r in parsed_regions if params['region'] in r['name']), None)
        if not region_data:
            print(f"⚠️ Регион '{params['region']}' для элемента {element} не найден.")
            continue
            
        # Переводим кинетическую энергию (KE) в энергию связи (BE)
        # Формула: BE = hν - KE - φ
        be = region_data['hv'] - region_data['ke'] - region_data['wf']
        counts = region_data['counts']
        trans = region_data['trans']
        
        # Коррекция на функцию пропускания анализатора (Transmission Function)
        trans_safe = np.where(trans == 0, 1e-10, trans)
        intensity = counts / trans_safe
        
        # Инвертируем массивы, чтобы BE возрастала (требуется для корректного интегрирования)
        be, intensity = be[::-1], intensity[::-1]
        
        # Выбираем только нужный диапазон энергий (границы пика)
        mask = (be >= params['be_min']) & (be <= params['be_max'])
        if not np.any(mask):
            print(f"⚠️ Границы для {element} вне диапазона данных.")
            continue
            
        be_int = be[mask]
        int_int = intensity[mask]
        
        # Вычисление и вычитание линейного фона
        bg_start, bg_end = int_int[0], int_int[-1]
        background = np.linspace(bg_start, bg_end, len(be_int))
        net_intensity = int_int - background
        
        # Расчет площади под пиком (интеграл)
        area = np.trapz(net_intensity, be_int)
        
        # Атомная чувствительность (Area / RSF)
        atomic_sensitivity = area / params['rsf']
        
        results.append({
            'Element': element, 'Peak': params['region'], 'Area': area,
            'RSF': params['rsf'], 'Sensitivity': atomic_sensitivity,
            'Include': params['include_in_total'],
            'be_plot': be_int, 'int_plot': int_int, 'bg_plot': background
        })
        
    # Нормировка на 100% (только для элементов образца, без адвентивного углерода)
    total_sensitivity = sum(r['Sensitivity'] for r in results if r['Include'])
    
    print("\n" + "="*75)
    print(f"{'Элемент':<8} {'Пик':<12} {'Площадь':<14} {'RSF':<6} {'Атомная %':<12} {'Примечание'}")
    print("-" * 75)
    
    for r in results:
        if r['Include']:
            conc = (r['Sensitivity'] / total_sensitivity) * 100 if total_sensitivity > 0 else 0
            note = ""
        else:
            conc = (r['Sensitivity'] / (total_sensitivity + r['Sensitivity'])) * 100
            note = "(Поверхностный)"
            
        r['Concentration'] = conc
        print(f"{r['Element']:<8} {r['Peak']:<12} {r['Area']:<14.2f} {r['RSF']:<6.2f} {conc:<12.2f}% {note}")
        
    print("="*75)
    print(f"{'БАЛАНС':<20} {'':<18} {total_sensitivity:<14.2f} 100.00%")
    print("="*75)
    
    return results

# ==========================================
# 4. ВИЗУАЛИЗАЦИЯ (ПРОВЕРКА ИНТЕГРИРОВАНИЯ)
# ==========================================
def plot_integration(results):
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4*n))
    if n == 1: axes = [axes]
        
    for ax, r in zip(axes, results):
        # График спектра и фона
        ax.plot(r['be_plot'], r['int_plot'], label='Спектр (с фоном)', color='blue', lw=1.5)
        ax.plot(r['be_plot'], r['bg_plot'], label='Линейный фон', color='red', linestyle='--')
        
        # Закрашиваем площадь, которая идет в расчет
        ax.fill_between(r['be_plot'], r['bg_plot'], r['int_plot'], color='green', alpha=0.3, label='Интегрируемая площадь')
        
        ax.set_title(f"{r['Element']} ({r['Peak']}) — {r['Concentration']:.2f} at.%", fontsize=12, fontweight='bold')
        ax.set_xlabel('Энергия связи (eV)')
        ax.set_ylabel('Интенсивность (Counts/s / Trans)')
        ax.invert_xaxis() # Традиция XPS: от большей энергии к меньшей
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('XPS_Concentrations_Analysis.png', dpi=300)
    plt.show()

# ==========================================
# ЗАПУСК СКРИПТА
# ==========================================
if __name__ == '__main__':
    # Укажите путь к вашему файлу (используйте r'' для Windows)
    filename = r'D:\Users\Lub\ExperimentalData\Data\NbN(5)GaN  t0 -Mg -EP20.xy'
    
    # Для демонстрации используем данные из контекста:
    #filename = 'Pasted_Text_1792063490589.txt' 
    
    try:
        print("Чтение файла и парсинг регионов...")
        regions = parse_specs_xps(filename)
        print(f"Найдено регионов: {len(regions)}")
        
        print("\nРасчет концентраций...")
        results = calculate_concentrations(regions, ELEMENT_CONFIG)
        
        print("\nПостроение графиков интегрирования...")
        plot_integration(results)
        
    except Exception as e:
        print(f"Ошибка при выполнении: {e}")