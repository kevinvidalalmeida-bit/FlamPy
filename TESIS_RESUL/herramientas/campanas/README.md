# Campañas reproducibles

Estos scripts reproducen las campañas conservadas de la tesis. Ejecútalos desde
la raíz del repositorio tras instalar el paquete en modo editable. Las rutas por
defecto apuntan a `TESIS_RESUL/corridas/`, que se mantiene localmente por el
volumen de los datos.

- `benchmark_flame_sweeps.py`: campaña L3 de llamas individuales.
- `benchmark_flame_mesh_sensitivity.py`: sensibilidad espacial de L3.
- `diagnose_flame_campaign.py`, `diagnose_flame_conditioning.py` y
  `diagnose_flame_recovery.py`: instrumentación y diagnósticos de la campaña.
- `extend_flame_transport_sweeps.py`: completa los cuatro transportes de la
  campaña L3.
- `benchmark_fgm_campaign.py`: construcción FGM de referencia.
- `benchmark_fgm_solver_comparison.py` y
  `benchmark_fgm_h2_solver_comparison.py`: comparación KFLAME--Cantera.
- `benchmark_fgm_h2_holdouts.py`: llamas independientes retenidas de H2.

Use `--dry-run` antes de lanzar una campaña y `--resume` únicamente sobre una
carpeta cuyo manifiesto coincida con el código y protocolo de la ejecución.
