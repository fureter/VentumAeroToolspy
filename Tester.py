from gnc.navigation.gnss.acq_engine import acq_iq_example

def main():
    # Search for a signal within ±50 chips in 25 steps increments within ±5000Hz in 25 steps.
    # Coherently sum across 10000 blocks to get a 40dB processing gain.
    acq_iq_example(coherent_integration_blocks=int(10**(40/10)), block_time=1E-3, css=(-50,50), css_steps=25,
                   fss=(-5E3, 5E3), fss_steps=25)

if "__main__" in __name__:
    main()
