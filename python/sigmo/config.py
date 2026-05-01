import dpctl

def get_default_queue():
    """Ritorna una coda SYCL sulla GPU se disponibile, altrimenti CPU."""
    try:
        device = dpctl.select_gpu_device() 
        return dpctl.SyclQueue(device)
    except:
        try:
            device = dpctl.select_cpu_device()
            return dpctl.SyclQueue(device)
        except:
            raise RuntimeError("Nessun dispositivo SYCL (GPU o CPU) trovato.")