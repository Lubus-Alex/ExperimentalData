import numpy as np
import matplotlib.pyplot as plt
import re

def plot_all_xps_regions(filename):
    """
    Читает файл SpecsLab2, парсит все регионы и строит графики 
    в шкале энергии связи (Binding Energy).
    """
    # Чтение содержимого файла
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Разделяем текст на блоки по маркеру начала нового региона
    blocks = re.split(r'# values in kinetic energy\n# Region:\s+', content)
    
    num_regions = len(blocks) - 1
    if num_regions == 0:
        print("Регионы не найдены. Проверьте формат файла.")
        return

    print(f"Найдено регионов для построения: {num_regions}")

    # Создаем полотно для графиков (каждый регион на отдельном подграфике)
    fig, axes = plt.subplots(num_regions, 1, figsize=(10, 5 * num_regions))
    if num_regions == 1:
        axes = [axes]
        
    for i, block in enumerate(blocks[1:]):
        # 1. Извлекаем название региона
        region_name = block.split('\n')[0].strip()
        
        # 2. Извлекаем метаданные (энергия возбуждения и работа выхода)
        hv_match = re.search(r'# Excitation Energy:\s+([\d.]+)', block)
        wf_match = re.search(r'# Eff. Workfunction:\s+([\d.]+)', block)
        
        hv = float(hv_match.group(1)) if hv_match else 1253.6  # По умолчанию Mg Kα
        wf = float(wf_match.group(1)) if wf_match else 4.34
        
        # 3. Парсим числовые данные (после строки ColumnLabels)
        data_lines = []
        in_data = False
        for line in block.split('\n'):
            if line.startswith('# ColumnLabels:'):
                in_data = True
                continue
            if in_data and line.strip() and not line.startswith('#'):
                data_lines.append(line.strip())
                
        ke, counts, trans = [], [], []
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    ke.append(float(parts[0]))
                    counts.append(float(parts[1]))
                    trans.append(float(parts[2]))
                except ValueError:
                    pass
                    
        ke = np.array(ke)
        counts = np.array(counts)
        trans = np.array(trans)
        
        # 4. Рассчитываем шкалу энергии связи (Binding Energy)
        # Формула XPS: BE = hν - KE - φ
        be = hv - ke - wf
        
        # 5. Применяем коррекцию на функцию пропускания (Transmission Function)
        trans_safe = np.where(trans == 0, 1e-10, trans) # Защита от деления на ноль
        intensity_corr = counts / trans_safe
        
        # 6. Строим график
        ax = axes[i]
        # Серый график - сырые данные, синий - скорректированные
        ax.plot(be, counts, label='Raw Counts/s (Сырые данные)', color='lightgray', linewidth=1)
        ax.plot(be, intensity_corr, label='Transmission Corrected (С коррекцией)', color='blue', linewidth=1.5)
        
        ax.set_xlabel('Энергия связи / Binding Energy (eV)', fontsize=12)
        ax.set_ylabel('Интенсивность / Intensity (Counts/s)', fontsize=12)
        ax.set_title(f'Регион: {region_name}', fontsize=14, fontweight='bold')
        
        # В XPS спектроскопии ось X традиционно инвертирована 
        # (от большей энергии связи к меньшей слева направо)
        ax.invert_xaxis()
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    
    # Сохранение и показ результата
    save_path = 'XPS_All_Regions_Plots.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Графики успешно сохранены в файл: {save_path}")
    plt.show()

# Запуск функции (укажите имя вашего файла)
# plot_all_xps_regions('Pasted_Text_1792063490589.txt')