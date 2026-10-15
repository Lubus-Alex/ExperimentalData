import numpy as np
import matplotlib.pyplot as plt
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

# =============================================================================
# DATA PARSER
# =============================================================================

@dataclass
class XPSRegion:
    """Stores metadata and data for one XPS region."""
    name: str
    analysis_method: str
    analyzer: str
    scan_mode: str
    num_scans: int
    values_per_curve: int
    dwell_time: float
    excitation_energy: float
    start_kinetic_energy: float
    pass_energy: float
    bias_voltage: float
    detector_voltage: float
    work_function: float
    kinetic_energy: np.ndarray
    counts_per_sec: np.ndarray
    transmission: np.ndarray

    @property
    def binding_energy(self) -> np.ndarray:
        """Convert kinetic energy to binding energy: BE = hν - KE - φ"""
        return self.excitation_energy - self.kinetic_energy - self.work_function


def parse_specs_xps(filename: str) -> List[XPSRegion]:
    """Parse a SpecsLab2 XY-Serializer Export file."""
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Split into regions by "# Region:" markers
    region_blocks = re.split(r'(?=# values in kinetic energy)', content)
    regions = []
    
    for block in region_blocks:
        if '# Region:' not in block:
            continue
        
        # Parse metadata
        def get_value(pattern, default=''):
            match = re.search(pattern, block)
            return match.group(1).strip() if match else default
        
        name = get_value(r'# Region:\s+(.+)')
        method = get_value(r'# Analysis Method:\s+(.+)')
        analyzer = get_value(r'# Analyzer:\s+(.+)')
        scan_mode = get_value(r'# Scan Mode:\s+(.+)')
        num_scans = int(get_value(r'# Number of Scans:\s+(\d+)', '1'))
        values_per_curve = int(get_value(r'# Values/Curve:\s+(\d+)', '0'))
        dwell_time = float(get_value(r'# Dwell Time:\s+([\d.]+)', '0'))
        excitation_energy = float(get_value(r'# Excitation Energy:\s+([\d.]+)', '0'))
        ke_start = float(get_value(r'# Kinetic Energy:\s+([\d.]+)', '0'))
        pass_energy = float(get_value(r'# Pass Energy:\s+([\d.]+)', '0'))
        bias_voltage = float(get_value(r'# Bias Voltage:\s+([\d.]+)', '0'))
        detector_voltage = float(get_value(r'# Detector Voltage:\s+([\d.]+)', '0'))
        work_function = float(get_value(r'# Eff. Workfunction:\s+([\d.]+)', '0'))
        
        # Parse numerical data (after ColumnLabels)
        data_section = block.split('# ColumnLabels:')[-1] if '# ColumnLabels:' in block else ''
        
        energies, counts, transmissions = [], [], []
        for line in data_section.strip().split('\n'):
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    energies.append(float(parts[0]))
                    counts.append(float(parts[1]))
                    transmissions.append(float(parts[2]))
                except ValueError:
                    continue
        
        if energies:
            region = XPSRegion(
                name=name,
                analysis_method=method,
                analyzer=analyzer,
                scan_mode=scan_mode,
                num_scans=num_scans,
                values_per_curve=values_per_curve,
                dwell_time=dwell_time,
                excitation_energy=excitation_energy,
                start_kinetic_energy=ke_start,
                pass_energy=pass_energy,
                bias_voltage=bias_voltage,
                detector_voltage=detector_voltage,
                work_function=work_function,
                kinetic_energy=np.array(energies),
                counts_per_sec=np.array(counts),
                transmission=np.array(transmissions)
            )
            regions.append(region)
    
    return regions


# =============================================================================
# DATA PROCESSING
# =============================================================================

def transmission_correction(region: XPSRegion) -> np.ndarray:
    """Apply transmission function correction to counts."""
    return region.counts_per_sec / region.transmission


def shirley_background(spectrum: np.ndarray, energy: np.ndarray, 
                       n_iter: int = 100) -> np.ndarray:
    """Calculate Shirley-type background."""
    background = np.full_like(spectrum, spectrum[0])
    
    for _ in range(n_iter):
        integral = np.zeros_like(energy)
        for i in range(len(energy)):
            integral[i] = np.trapz(background[:i+1], energy[:i+1]) if i > 0 else 0
        
        total_integral = np.trapz(background, energy)
        bg_new = spectrum[0] + (spectrum[-1] - spectrum[0]) * integral / total_integral
        if np.allclose(bg_new, background, atol=1e-6):
            break
        background = bg_new
    
    return background


def smooth_spectrum(counts: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply simple moving average smoothing."""
    if window <= 1:
        return counts
    kernel = np.ones(window) / window
    smoothed = np.convolve(counts, kernel, mode='same')
    # Handle edges
    smoothed[:window//2] = counts[:window//2]
    smoothed[-(window//2):] = counts[-(window//2):]
    return smoothed


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_survey_spectrum(region: XPSRegion, save_path: str = None):
    """Plot the wide/survey spectrum in Binding Energy scale."""
    be = region.binding_energy
    intensity = transmission_correction(region)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(be, intensity, 'k-', linewidth=0.8)
    ax.set_xlabel('Binding Energy (eV)', fontsize=12)
    ax.set_ylabel('Intensity (arb. units)', fontsize=12)
    ax.set_title(f'Survey Spectrum - {region.name}', fontsize=14)
    ax.set_xlim(max(be), min(be))  # XPS convention: reversed x-axis
    
    # Annotate expected peaks
    annotations = {
        'Nb 3d': 202.8, 'Ga 2p': 1117.0, 'N 1s': 397.0,
        'C 1s': 284.8, 'O 1s': 530.0, 'Ga 3p': 161.0,
        'Nb 3p': 362.0, 'Ga 3s': 229.0
    }
    
    for label, be_pos in annotations.items():
        if min(be) < be_pos < max(be):
            ax.axvline(x=be_pos, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.text(be_pos, ax.get_ylim()[1]*0.95, label, fontsize=8,
                   ha='center', va='top', rotation=90)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_high_resolution(regions: List[XPSRegion], target_regions: List[str], 
                         save_path: str = None):
    """Plot high-resolution spectra for specified regions."""
    n_plots = len(target_regions)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]
    
    for ax, target in zip(axes, target_regions):
        for region in regions:
            if target.lower() in region.name.lower():
                be = region.binding_energy
                intensity = transmission_correction(region)
                
                ax.plot(be, intensity, 'b-', linewidth=1.2, label='Raw Data')
                ax.set_xlabel('Binding Energy (eV)', fontsize=11)
                ax.set_ylabel('Intensity (arb. units)', fontsize=11)
                ax.set_title(f'{region.name} (Pass Energy: {region.pass_energy} eV, '
                           f'Scans: {region.num_scans})', fontsize=12)
                ax.set_xlim(max(be), min(be))
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
                break
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_all_regions_overlay(regions: List[XPSRegion], save_path: str = None):
    """Plot all regions on binding energy scale with transmission correction."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(regions)))
    
    for region, color in zip(regions, colors):
        if 'Wide' in region.name:
            continue  # Skip survey for clarity
        be = region.binding_energy
        intensity = transmission_correction(region)
        ax.plot(be, intensity, color=color, linewidth=1.0, label=region.name)
    
    ax.set_xlabel('Binding Energy (eV)', fontsize=12)
    ax.set_ylabel('Intensity (arb. units)', fontsize=12)
    ax.set_title('XPS - 5nm Nb on GaN (All High-Resolution Regions)', fontsize=14)
    ax.set_xlim(1250, 0)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Parse the data
    filename = r'D:\Users\Lub\ExperimentalData\Data\NbN(5)GaN  t0 -Mg -EP20.xy'
    
    # For demonstration, using embedded data parsing
    # If file is available:
    try:
        regions = parse_specs_xps(filename)
    except FileNotFoundError:
        print(f"File '{filename}' not found. Please provide the correct path.")
        print("Alternatively, paste the data or provide the file path.")
        exit()
    
    # Print summary
    print("=" * 70)
    print("XPS DATA SUMMARY")
    print("=" * 70)
    print(f"{'Region':<20} {'Scans':<6} {'Pass E':<8} {'KE Range':<16} {'BE Range'}")
    print("-" * 70)
    for r in regions:
        be_min, be_max = r.binding_energy.min(), r.binding_energy.max()
        print(f"{r.name:<20} {r.num_scans:<6} {r.pass_energy:<8.0f} "
              f"{r.kinetic_energy[0]:.1f}-{r.kinetic_energy[-1]:.1f}  "
              f"{be_max:.1f}-{be_min:.1f} eV")
    print("=" * 70)
    
    # Plot survey spectrum
    survey = [r for r in regions if 'Wide' in r.name]
    if survey:
        plot_survey_spectrum(survey[0], save_path='survey_spectrum.png')
    
    # Plot high-resolution spectra
    hr_targets = ['Nb 3d', 'O 1s', 'C 1s', 'N 1s', 'Ga 2p', 'Na 1s']
    plot_high_resolution(regions, hr_targets, save_path='high_resolution.png')
    
    # Plot all regions
    plot_all_regions_overlay(regions, save_path='all_regions.png')
    
    # Export data to CSV (optional)
    for region in regions:
        safe_name = region.name.replace(' ', '_').replace('/', '_')
        be = region.binding_energy
        corrected = transmission_correction(region)
        
        df_data = np.column_stack([
            region.kinetic_energy,
            be,
            region.counts_per_sec,
            region.transmission,
            corrected
        ])
        header = 'Kinetic_Energy(eV),Binding_Energy(eV),Counts_per_s,Transmission,Corrected_Intensity'
        np.savetxt(f'{safe_name}.csv', df_data, delimiter=',', header=header, comments='')
        print(f"Exported: {safe_name}.csv")
    
    print("\nDone! All plots and CSV files generated.")