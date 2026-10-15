# -*- coding: utf-8 -*-
"""ООП-конвейер количественного анализа XPS-данных SpecsLab2."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# УТИЛИТЫ ОБРАБОТКИ СИГНАЛА
# ============================================================
class SignalUtils:
    """Статические помощники: сглаживание и оценка шума."""

    @staticmethod
    def moving_average(y: np.ndarray, window: int = 5) -> np.ndarray:
        if window <= 1:
            return y.copy()
        out = np.convolve(y, np.ones(window) / window, mode='same')
        out[:window // 2] = y[:window // 2]
        out[-(window // 2):] = y[-(window // 2):]
        return out

    @staticmethod
    def noise_sigma(y: np.ndarray, window: int = 5) -> float:
        return float(np.std(y - SignalUtils.moving_average(y, window)))


# ============================================================
# МОДЕЛЬ ДАННЫХ
# ============================================================
@dataclass
class XPSRegion:
    """Один регион спектра (метаданные + массивы)."""
    name: str
    kinetic_energy: np.ndarray
    counts_per_sec: np.ndarray
    transmission: np.ndarray
    excitation_energy: float = 1253.6
    work_function: float = 4.34
    pass_energy: float = np.nan
    num_scans: int = 1

    @property
    def binding_energy(self) -> np.ndarray:
        """BE = hν − KE − φ."""
        return self.excitation_energy - self.kinetic_energy - self.work_function

    @property
    def intensity(self) -> np.ndarray:
        """Интенсивность с коррекцией функции пропускания."""
        tr = np.where(self.transmission == 0, 1e-10, self.transmission)
        return self.counts_per_sec / tr

    def be_sorted(self) -> Tuple[np.ndarray, np.ndarray]:
        """(BE по возрастанию, интенсивность)."""
        idx = np.argsort(self.binding_energy)
        return self.binding_energy[idx], self.intensity[idx]

    def apply_energy_shift(self, shift: float) -> None:
        """Калибровка шкалы: BE_new = BE + shift (через work_function)."""
        self.work_function -= shift

    def __repr__(self) -> str:
        return f"<XPSRegion '{self.name}', {self.kinetic_energy.size} точек>"


class XPSDataset:
    """Коллекция регионов с поиском по имени и общей калибровкой."""

    def __init__(self, regions: Optional[List[XPSRegion]] = None, group: str = ''):
        self.regions: List[XPSRegion] = regions or []
        self.group = group

    def __len__(self) -> int:
        return len(self.regions)

    def __iter__(self) -> Iterator[XPSRegion]:
        return iter(self.regions)

    def add(self, region: XPSRegion) -> None:
        self.regions.append(region)

    def find(self, name_part: str) -> XPSRegion:
        for r in self.regions:
            if name_part.lower() in r.name.lower():
                return r
        raise KeyError(f"Регион '{name_part}' не найден. Есть: {[r.name for r in self]}")

    def apply_charge_shift(self, shift: float) -> None:
        for r in self:
            r.apply_energy_shift(shift)


# ============================================================
# ПАРСЕР (Factory для XPSDataset)
# ============================================================
class SpecsLab2Parser:
    """Читает XY-Serializer Export и строит XPSDataset."""

    _SEP = re.compile(r'# values in kinetic energy\s*\n# Region:\s+')
    _META = {
        'excitation_energy': r'# Excitation Energy:\s*([\d.eE+-]+)',
        'work_function':     r'# Eff\. Workfunction:\s*([\d.eE+-]+)',
        'pass_energy':       r'# Pass Energy:\s*([\d.eE+-]+)',
        'num_scans':         r'# Number of Scans:\s*(\d+)',
    }

    def __init__(self, filename: str):
        self.filename = filename

    def parse(self) -> XPSDataset:
        with open(self.filename, 'r', encoding='utf-8') as fh:
            content = fh.read()
        m = re.search(r'# Group:\s*(.+)', content)
        dataset = XPSDataset(group=m.group(1).strip() if m else '')
        for block in self._SEP.split(content)[1:]:
            dataset.add(self._parse_region(block))
        return dataset

    @classmethod
    def _parse_region(cls, block: str) -> XPSRegion:
        meta = {}
        for key, pat in cls._META.items():
            m = re.search(pat, block)
            meta[key] = m.group(1) if m else None
        ke, cps, tr = cls._parse_columns(block)
        return XPSRegion(
            name=block.split('\n', 1)[0].strip(),
            kinetic_energy=ke, counts_per_sec=cps, transmission=tr,
            excitation_energy=float(meta['excitation_energy'] or 1253.6),
            work_function=float(meta['work_function'] or 4.34),
            pass_energy=float(meta['pass_energy']) if meta['pass_energy'] else np.nan,
            num_scans=int(meta['num_scans'] or 1),
        )

    @staticmethod
    def _parse_columns(block: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ke, cps, tr = [], [], []
        for line in block.split('# ColumnLabels:')[-1].splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split()
            if len(p) >= 3:
                try:
                    ke.append(float(p[0])); cps.append(float(p[1])); tr.append(float(p[2]))
                except ValueError:
                    continue
        return np.array(ke), np.array(cps), np.array(tr)


# ============================================================
# ФОН: ПАТТЕРН STRATEGY + FACTORY
# ============================================================
class BackgroundModel(ABC):
    """Абстрактная модель фона. be должен возрастать."""
    name: str = 'base'

    @abstractmethod
    def compute(self, be: np.ndarray, y: np.ndarray) -> np.ndarray: ...


class LinearBackground(BackgroundModel):
    name = 'linear'

    def compute(self, be: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linspace(y[0], y[-1], y.size)


class ShirleyBackground(BackgroundModel):
    """Итеративный Ширли; авто-откат на линейный для убывающего фона."""
    name = 'shirley'

    def __init__(self, n_iter: int = 200, tol: float = 1e-6):
        self.n_iter, self.tol = n_iter, tol

    def compute(self, be: np.ndarray, y: np.ndarray) -> np.ndarray:
        if y[-1] <= y[0]:
            return LinearBackground().compute(be, y)
        b = np.full_like(y, y[0])
        for _ in range(self.n_iter):
            net = np.clip(y - b, 0.0, None)
            cum = np.zeros_like(y)
            cum[1:] = np.cumsum((net[1:] + net[:-1]) * 0.5 * np.diff(be))
            if cum[-1] <= 0:
                return LinearBackground().compute(be, y)
            b_new = y[0] + (y[-1] - y[0]) * cum / cum[-1]
            if np.max(np.abs(b_new - b)) < self.tol * max(1.0, y.max()):
                return b_new
            b = b_new
        return b


class BackgroundFactory:
    _registry = {'linear': LinearBackground, 'shirley': ShirleyBackground}

    @classmethod
    def create(cls, name: str, **kwargs) -> BackgroundModel:
        try:
            return cls._registry[name.lower()](**kwargs)
        except KeyError:
            raise ValueError(f'Неизвестный фон "{name}". Доступны: {list(cls._registry)}')

    @classmethod
    def register(cls, name: str, klass: type) -> None:
        """Расширение: BackgroundFactory.register('tougaard', TougaardBackground)."""
        cls._registry[name.lower()] = klass


# ============================================================
# НАСТРОЙКИ ЭЛЕМЕНТОВ И РЕЗУЛЬТАТ ПИКА
# ============================================================
@dataclass(frozen=True)
class ElementSettings:
    symbol: str
    region: str
    search: Tuple[float, float]      # окно поиска максимума (BE после калибровки)
    offsets: Tuple[float, float]     # границы интегрирования относительно центра
    rsf: float
    include_in_total: bool = True


DEFAULT_ELEMENTS: Tuple[ElementSettings, ...] = (
    ElementSettings('O',  'O 1s',       (527, 536),   (-4.0, 4.0),  2.93),
    ElementSettings('N',  'N 1s Nb 3p', (395, 401),   (-3.0, 3.0),  1.80),
    ElementSettings('Nb', 'Nb 3d',      (203, 212),   (-3.2, 6.0),  3.00),
    ElementSettings('Na', 'Na 1s',      (1069, 1077), (-3.5, 3.5),  2.08),
    ElementSettings('Ga', 'Ga 2p',      (1112, 1125), (-3.0, 31.0), 5.96),
    ElementSettings('C',  'C 1s',       (282, 290),   (-3.5, 3.5),  1.00,
                    include_in_total=False),
)


@dataclass
class PeakResult:
    settings: ElementSettings
    center: float
    area: float
    below_detection_limit: bool
    be: np.ndarray
    y: np.ndarray
    background: np.ndarray
    net: np.ndarray                  # y − bg, обрезано снизу нулём
    background_name: str

    @property
    def symbol(self) -> str:
        return self.settings.symbol

    @property
    def sensitivity(self) -> float:
        return self.area / self.settings.rsf


# ============================================================
# КВАНТИФИКАЦИЯ
# ============================================================
class XPSQuantifier:
    """Поиск пиков, вычитание фона, площади, предел обнаружения."""

    def __init__(self, dataset: XPSDataset,
                 elements: Tuple[ElementSettings, ...] = DEFAULT_ELEMENTS,
                 background: BackgroundModel | str = 'shirley',
                 calibrate_c1s: Optional[float] = 284.8,
                 dl_factor: float = 3.0):
        self.dataset = dataset
        self.elements = list(elements)
        self.background = (BackgroundFactory.create(background)
                           if isinstance(background, str) else background)
        self.calibrate_c1s = calibrate_c1s
        self.dl_factor = dl_factor
        self.charge_shift = 0.0

    def run(self) -> 'XPSReport':
        self._calibrate()
        peaks = [self._analyze(cfg) for cfg in self.elements]
        return XPSReport(peaks, self)

    def _calibrate(self) -> None:
        self.charge_shift = 0.0
        if self.calibrate_c1s is None:
            return
        c_cfg = next((e for e in self.elements if e.symbol == 'C'), None)
        if c_cfg is None:
            return
        be, y = self.dataset.find(c_cfg.region).be_sorted()
        m = (be > c_cfg.search[0]) & (be < c_cfg.search[1])
        if not m.any():
            return
        peak_be = be[m][np.argmax(SignalUtils.moving_average(y[m]))]
        self.charge_shift = self.calibrate_c1s - peak_be
        self.dataset.apply_charge_shift(self.charge_shift)

    def _analyze(self, cfg: ElementSettings) -> PeakResult:
        be, y = self.dataset.find(cfg.region).be_sorted()
        m = (be > cfg.search[0]) & (be < cfg.search[1])
        idx = np.where(m)[0][np.argmax(SignalUtils.moving_average(y[m]))]
        center = float(be[idx])
        w = (be >= center + cfg.offsets[0]) & (be <= center + cfg.offsets[1])
        be_w, y_w = be[w], y[w]
        bg = self.background.compute(be_w, y_w)
        net = np.clip(y_w - bg, 0.0, None)          # площади ≥ 0 гарантированно
        below_dl = bool(net.max() < self.dl_factor * SignalUtils.noise_sigma(y_w))
        area = 0.0 if below_dl else float(np.trapz(net, be_w))
        return PeakResult(cfg, center, area, below_dl, be_w, y_w, bg, net,
                          self.background.name)


# ============================================================
# ОТЧЕТ
# ============================================================
class XPSReport:
    """Таблица концентраций и доступ к пикам."""

    def __init__(self, peaks: List[PeakResult], quantifier: XPSQuantifier):
        self.peaks = peaks
        self.charge_shift = quantifier.charge_shift
        self.background_name = quantifier.background.name
        self._total = sum(p.sensitivity for p in peaks
                          if p.settings.include_in_total and p.area > 0)

    def concentration(self, peak: PeakResult) -> float:
        if peak.below_detection_limit or self._total <= 0:
            return 0.0
        if peak.settings.include_in_total:
            return peak.sensitivity / self._total * 100.0
        return peak.sensitivity / (self._total + peak.sensitivity) * 100.0

    def as_table(self) -> str:
        L = ['=' * 86,
             f'Фон: {self.background_name}    Сдвиг калибровки (C 1s): '
             f'{self.charge_shift:+.2f} эВ',
             '-' * 86,
             f"{'Эл.':<5}{'Пик, эВ':>10}{'Площадь':>13}{'RSF':>7}{'at.%':>9}   Примечание",
             '-' * 86]
        for p in self.peaks:
            note = ('< предела обнаружения' if p.below_detection_limit
                    else ('' if p.settings.include_in_total else 'адвентивный, вне баланса'))
            L.append(f'{p.symbol:<5}{p.center:>10.2f}{p.area:>13.1f}'
                     f'{p.settings.rsf:>7.2f}{self.concentration(p):>9.2f}   {note}')
        L.append('=' * 86)
        return '\n'.join(L)

    def to_csv(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('element;peak_BE_eV;area;rsf;at_percent;below_DL\n')
            for p in self.peaks:
                fh.write(f'{p.symbol};{p.center:.3f};{p.area:.2f};'
                         f'{p.settings.rsf:.3f};{self.concentration(p):.3f};'
                         f'{int(p.below_detection_limit)}\n')

    def __str__(self) -> str:
        return self.as_table()


# ============================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================
class XPSPlotter:
    def __init__(self, dataset: XPSDataset):
        self.dataset = dataset

    def plot_regions(self, save_path: Optional[str] = None):
        regs = self.dataset.regions
        fig, axes = plt.subplots(len(regs), 1, figsize=(9, 3.0 * len(regs)),
                                 constrained_layout=True)
        for ax, r in zip(np.atleast_1d(axes), regs):
            be, y = r.be_sorted()
            ax.plot(be, y, 'k-', lw=0.9)
            ax.set_title(f'{r.name}  (EP={r.pass_energy:.0f} эВ, сканов={r.num_scans})')
            ax.set_xlim(be.max(), be.min())
            ax.set_ylabel('Counts/s / T'); ax.grid(alpha=0.3)
        np.atleast_1d(axes)[-1].set_xlabel('Binding Energy (eV)')
        if save_path:
            fig.savefig(save_path, dpi=300)
        return fig

    def plot_quantification(self, report: XPSReport, save_path: Optional[str] = None):
        peaks = report.peaks
        fig, axes = plt.subplots(len(peaks), 1, figsize=(9, 3.2 * len(peaks)),
                                 constrained_layout=True)
        for ax, p in zip(np.atleast_1d(axes), peaks):
            ax.plot(p.be, p.y, 'b-', lw=1.1, label='Спектр')
            ax.plot(p.be, p.background, 'r--', lw=1.1, label=f'Фон ({p.background_name})')
            ax.fill_between(p.be, p.background, p.background + p.net,
                            color='green', alpha=0.3, label='Площадь (≥0)')
            title = f'{p.symbol} — пик {p.center:.2f} эВ, {report.concentration(p):.2f} at.%'
            if p.below_detection_limit:
                title += '   [< предела обнаружения]'
            ax.set_title(title, fontweight='bold')
            ax.set_xlim(p.be.max(), p.be.min())
            ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.3)
        np.atleast_1d(axes)[-1].set_xlabel('Binding Energy (eV)')
        if save_path:
            fig.savefig(save_path, dpi=300)
        return fig


# ============================================================
# ФАСАД
# ============================================================
class XPSAnalysis:
    """Единая точка входа: парсинг → квантификация → графики."""

    def __init__(self, filename: str, background: str = 'shirley',
                 calibrate_c1s: Optional[float] = 284.8):
        self.dataset = SpecsLab2Parser(filename).parse()
        self.quantifier = XPSQuantifier(self.dataset, background=background,
                                        calibrate_c1s=calibrate_c1s)
        self.plotter = XPSPlotter(self.dataset)
        self.report: Optional[XPSReport] = None

    def quantify(self) -> XPSReport:
        self.report = self.quantifier.run()
        return self.report

    def plot_regions(self, save_path=None):
        return self.plotter.plot_regions(save_path)

    def plot_quantification(self, save_path=None):
        if self.report is None:
            self.quantify()
        return self.plotter.plot_quantification(self.report, save_path)


# ============================================================
# ТОЧКА ВХОДА
# ============================================================
if __name__ == '__main__':
    path = r'D:\Users\Lub\ExperimentalData\Data\NbN(5)GaN  t0 -Mg -EP20.xy'

    analysis = XPSAnalysis(path, background='shirley', calibrate_c1s=284.8)
    print(f'Группа: {analysis.dataset.group}, регионов: {len(analysis.dataset)}')

    report = analysis.quantify()
    print(report)                       # таблица at.%
    report.to_csv('xps_quant.csv')

    fig1 = analysis.plot_regions('xps_regions.png')
    fig2 = analysis.plot_quantification('xps_quant.png')
    plt.show()