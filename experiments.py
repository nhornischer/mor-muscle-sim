import base,solver,decomp

import matplotlib.pyplot as plt

# =============================================================================
# Call to Main
# =============================================================================
if __name__ == '__main__':
    """This script includes all the experiments and the corresponding commands conducted for the thesis, the resulting hashes are defined in "experiment_logs.txt"
    """

    # =============================================================================
    # FOM 
    # =============================================================================
    # Compute high dimensional basis solution
    hash_fom_basis,_=solver.main(22, 1e-3, 'FOM', 'EE', ['standard'])

    hash_fom_init,_=solver.main(1,1e-3,'FOM','EE',['standard'])
    
    # Calculate solutions for separated and overlapping waves with different time-step widhts [extreme, high, medium, low, min]= [1e-3,5e-3,1e-2,5e-2,1e-1]
    FOM_time_discr=[1e-3,1.2e-3,1.4e-3,1.6e-3,1.8e-3]
    
    # Separated
    hash_fom_sep_ext,_=solver.main(22, FOM_time_discr[0], 'FOM', 'EE', ['extract', hash_fom_basis, 18.0])
    hash_fom_sep_hig,_=solver.main(22, FOM_time_discr[1], 'FOM', 'EE', ['extract', hash_fom_basis, 18.0])
    hash_fom_sep_med,_=solver.main(22, FOM_time_discr[2], 'FOM', 'EE', ['extract', hash_fom_basis, 18.0])
    hash_fom_sep_low,_=solver.main(22, FOM_time_discr[3], 'FOM', 'EE', ['extract', hash_fom_basis, 18.0])
    hash_fom_sep_min,_=solver.main(22, FOM_time_discr[4], 'FOM', 'EE', ['extract', hash_fom_basis, 18.0])

    # Overlapping
    hash_fom_over_ext,_=solver.main(22, FOM_time_discr[0], 'FOM', 'EE', ['extract',hash_fom_basis, 5.0])
    hash_fom_over_hig,_=solver.main(22, FOM_time_discr[1], 'FOM', 'EE', ['extract',hash_fom_basis, 5.0])
    hash_fom_over_med,_=solver.main(22, FOM_time_discr[2], 'FOM', 'EE', ['extract',hash_fom_basis, 5.0])
    hash_fom_over_low,_=solver.main(22, FOM_time_discr[3], 'FOM', 'EE', ['extract',hash_fom_basis, 5.0])
    hash_fom_over_min,_=solver.main(22, FOM_time_discr[4], 'FOM', 'EE', ['extract',hash_fom_basis, 5.0])

    # =============================================================================
    # Reduced Basis
    # =============================================================================

    # Initialization of the wave with shift
    hash_dec_init=decomp.main(hash_fom_basis,2,True,'Simple')

    # Separated waves not shifted initialized with snapshots
    hash_dec_sep_ns=decomp.main(hash_fom_sep_ext,2,False,'dualSnapshots')

    # Separated waves initialized with snapshots
    hash_dec_sep=decomp.main(hash_fom_sep_ext,2,True,'dualSnapshots')

    # Separated waves initialized with snapshots with 4 basis functions
    hash_dec_sep_4=decomp.main(hash_fom_sep_ext,4,True,'dualSnapshots')

    # Separated waves simply initialized with [1,2,3,4] basis functions
    hash_dec_sep_simp_1=decomp.main(hash_fom_sep_ext,1,True,'Simple')
    hash_dec_sep_simp_2=decomp.main(hash_fom_sep_ext,2,True,'Simple')
    hash_dec_sep_simp_3=decomp.main(hash_fom_sep_ext,3,True,'Simple')
    hash_dec_sep_simp_4=decomp.main(hash_fom_sep_ext,4,True,'Simple')

    # Separated waves simply initialized with 2 basis functions for the left and right wave
    hash_dec_sep_simp_left=decomp.main(hash_fom_sep_ext,1,True,'SimpleLeft',data_wave='Left')
    hash_dec_sep_simp_right=decomp.main(hash_fom_sep_ext,1,True,'SimpleRight',data_wave='Right')
    hash_merged_sep = base.merge_spaces(hash_dec_sep_simp_left,hash_dec_sep_simp_right)

    # Separated waves initialized gauss with [2,4] basis functions
    hash_dec_sep_gaus_2=decomp.main(hash_fom_sep_ext,2,True,'gausDual')
    hash_dec_sep_gaus_4=decomp.main(hash_fom_sep_ext,4,True,'gausDual')

    # Overlapping waves not shifted initialized with snapshots
    hash_dec_over_ns=decomp.main(hash_fom_over_ext,2,False,'dualSnapshots')

    # Overlapping waves initialized with snapshots
    hash_dec_over=decomp.main(hash_fom_over_ext,2,True,'dualSnapshots')

    # Overlapping waves initialized with snapshots with 4 basis functions
    hash_dec_over_4=decomp.main(hash_fom_over_ext,4,True,'dualSnapshots')

    # Overlapping waves simply initialized with [1,2,3,4] basis functions
    hash_dec_over_simp_1=decomp.main(hash_fom_over_ext,1,True,'Simple')
    hash_dec_over_simp_2=decomp.main(hash_fom_over_ext,2,True,'Simple')
    hash_dec_over_simp_3=decomp.main(hash_fom_over_ext,3,True,'Simple')
    hash_dec_over_simp_4=decomp.main(hash_fom_over_ext,4,True,'Simple')

    # Overlapping waves simply initialized with 2 basis functions for the left and right wave
    hash_dec_over_simp_left=decomp.main(hash_fom_over_ext,1,True,'SimpleLeft',data_wave='Left')
    hash_dec_over_simp_right=decomp.main(hash_fom_over_ext,1,True,'SimpleRight',data_wave='Right')
    hash_merged_over = base.merge_spaces(hash_dec_over_simp_left,hash_dec_over_simp_right)

    # Overlapping waves initialized gauss with [2,4] basis functions
    hash_dec_over_gaus_2=decomp.main(hash_fom_over_ext,2,True,'gausDual')
    hash_dec_over_gaus_4=decomp.main(hash_fom_over_ext,4,True,'gausDual')

    # =============================================================================
    # ROM
    # =============================================================================

    # Different time-step widths
    ROM_time_discr=[1e-3,5e-3,1e-2,5e-2,1e-1]

    # Separated reduced based on snapshot
    hash_rom_sep_ext,_=solver.main(22, ROM_time_discr[0], 'ROM', 'EE', ['extract', hash_fom_basis, 18.0],hash_dec_sep)
    hash_rom_sep_hig,_=solver.main(22, ROM_time_discr[1], 'ROM', 'EE', ['extract', hash_fom_basis, 18.0],hash_dec_sep)
    hash_rom_sep_med,_=solver.main(22, ROM_time_discr[2], 'ROM', 'EE', ['extract', hash_fom_basis, 18.0],hash_dec_sep)
    hash_rom_sep_low,_=solver.main(22, ROM_time_discr[3], 'ROM', 'EE', ['extract', hash_fom_basis, 18.0],hash_dec_sep)
    hash_rom_sep_min,_=solver.main(22, ROM_time_discr[4], 'ROM', 'EE', ['extract', hash_fom_basis, 18.0],hash_dec_sep)

    # Overlapping
    hash_rom_over_ext,_=solver.main(22, ROM_time_discr[0], 'ROM', 'EE', ['extract',hash_fom_basis, 5.0],hash_dec_over)
    hash_rom_over_hig,_=solver.main(22, ROM_time_discr[1], 'ROM', 'EE', ['extract',hash_fom_basis, 5.0],hash_dec_over)
    hash_rom_over_med,_=solver.main(22, ROM_time_discr[2], 'ROM', 'EE', ['extract',hash_fom_basis, 5.0],hash_dec_over)
    hash_rom_over_low,_=solver.main(22, ROM_time_discr[3], 'ROM', 'EE', ['extract',hash_fom_basis, 5.0],hash_dec_over)
    hash_rom_over_min,_=solver.main(22, ROM_time_discr[4], 'ROM', 'EE', ['extract',hash_fom_basis, 5.0],hash_dec_over)


    # =============================================================================
    # Evaluation
    # =============================================================================

    # Evaluate FOM solution for different time-step widhts for separated
    base.calculate_error(hash_fom_sep_hig,hash_fom_basis)
    base.calculate_error(hash_fom_sep_med,hash_fom_basis)
    base.calculate_error(hash_fom_sep_low,hash_fom_basis)
    base.calculate_error(hash_fom_sep_min,hash_fom_basis)

    # Evaluate FOM solution for different time-step widhts for overlaping
    base.calculate_error(hash_fom_over_hig,hash_fom_basis)
    base.calculate_error(hash_fom_over_med,hash_fom_basis)
    base.calculate_error(hash_fom_over_low,hash_fom_basis)
    base.calculate_error(hash_fom_over_min,hash_fom_basis)

    # # Evaluate DECOMP for init
    base.calculate_error(hash_dec_init,hash_fom_basis)

    # Evaluate DECOMP for complete separated snapshot
    base.calculate_error(hash_dec_sep,hash_fom_sep_ext)

    # Evaluate DECOMP with no shift
    base.calculate_error(hash_dec_sep_ns,hash_fom_sep_ext)
    base.calculate_error(hash_dec_over_ns,hash_fom_over_ext)

    # Evaluate DECOMP with shift and initialized with snapshots
    base.calculate_error(hash_dec_sep,hash_fom_sep_ext)
    base.calculate_error(hash_dec_sep_4,hash_fom_sep_ext)
    base.calculate_error(hash_dec_over,hash_fom_over_ext)
    base.calculate_error(hash_dec_over_4,hash_fom_over_ext)

    # Evaluate DECOMP with shift and initialized simply
    base.calculate_error(hash_dec_sep_simp_1,hash_fom_sep_ext)
    base.calculate_error(hash_dec_sep_simp_2,hash_fom_sep_ext)
    base.calculate_error(hash_dec_sep_simp_3,hash_fom_sep_ext)
    base.calculate_error(hash_dec_sep_simp_4,hash_fom_sep_ext)
    base.calculate_error(hash_dec_over_simp_1,hash_fom_over_ext)
    base.calculate_error(hash_dec_over_simp_2,hash_fom_over_ext)
    base.calculate_error(hash_dec_over_simp_3,hash_fom_over_ext)
    base.calculate_error(hash_dec_over_simp_4,hash_fom_over_ext)

    # Evaluate DECOMP with shift and initialized simply for left and right waves
    base.calculate_error(hash_dec_sep_simp_left,hash_fom_sep_ext)
    base.calculate_error(hash_dec_sep_simp_right,hash_fom_sep_ext)
    base.calculate_error(hash_dec_over_simp_left,hash_fom_over_ext)
    base.calculate_error(hash_dec_over_simp_right,hash_fom_over_ext)
    base.calculate_error(hash_merged_sep,hash_fom_sep_ext)
    base.calculate_error(hash_merged_over,hash_fom_over_ext)


    # Evaluate DECOMP with shift and initialized with Gaus
    base.calculate_error(hash_dec_sep_gaus_2,hash_fom_sep_ext)
    base.calculate_error(hash_dec_sep_gaus_4,hash_fom_sep_ext)
    base.calculate_error(hash_dec_over_gaus_2,hash_fom_over_ext)
    base.calculate_error(hash_dec_over_gaus_4,hash_fom_over_ext)

    # Evaluate ROM for separated waves
    base.calculate_error(hash_rom_sep_ext,hash_fom_basis)
    base.calculate_error(hash_rom_sep_hig,hash_fom_basis)
    base.calculate_error(hash_rom_sep_med,hash_fom_basis)
    base.calculate_error(hash_rom_sep_low,hash_fom_basis)
    base.calculate_error(hash_rom_sep_min,hash_fom_basis)
    

    # Evaluate ROM for overlapping waves
    base.calculate_error(hash_rom_over_ext,hash_fom_basis)
    base.calculate_error(hash_rom_over_hig,hash_fom_basis)
    base.calculate_error(hash_rom_over_med,hash_fom_basis)
    base.calculate_error(hash_rom_over_low,hash_fom_basis)
    base.calculate_error(hash_rom_over_min,hash_fom_basis)


    # Time test for FOM
    hash_time_FOM_sep,_=base.time_test(22, 1e-3, 'FOM', 'EE', ['extract',hash_fom_basis, 18.0])
    hash_time_FOM_over,_=base.time_test(22, 1e-3, 'FOM', 'EE', ['extract',hash_fom_basis, 5.0])

    # Time test for ROM
    hash_time_sep_ext,_=base.time_test(22, ROM_time_discr[0], 'ROM', 'EE', ['extract',hash_fom_basis, 18.0],hash_dec_sep)
    hash_time_sep_hig,_=base.time_test(22, ROM_time_discr[1], 'ROM', 'EE', ['extract',hash_fom_basis, 18.0],hash_dec_sep)
    hash_time_sep_med,_=base.time_test(22, ROM_time_discr[2], 'ROM', 'EE', ['extract',hash_fom_basis, 18.0],hash_dec_sep)
    hash_time_sep_low,_=base.time_test(22, ROM_time_discr[3], 'ROM', 'EE', ['extract',hash_fom_basis, 18.0],hash_dec_sep)

    # Time test for ROM
    hash_time_over_ext,_=base.time_test(22, ROM_time_discr[0], 'ROM', 'EE', ['extract',hash_fom_basis, 5.0],hash_dec_over)
    hash_time_over_hig,_=base.time_test(22, ROM_time_discr[1], 'ROM', 'EE', ['extract',hash_fom_basis, 5.0],hash_dec_over)
    hash_time_over_med,_=base.time_test(22, ROM_time_discr[2], 'ROM', 'EE', ['extract',hash_fom_basis, 5.0],hash_dec_over)
    hash_time_over_low,_=base.time_test(22, ROM_time_discr[3], 'ROM', 'EE', ['extract',hash_fom_basis, 5.0],hash_dec_over)

    # =============================================================================
    # Plots
    # =============================================================================  
    
    # Plot FOM with basis solution
    base.plot_states(hash_fom_basis)
    base.plot_snapshots(hash_fom_basis)

    # Plot FOM init with basis solution
    base.plot_states(hash_fom_init)
    base.plot_snapshots(hash_fom_init)
    
    # Plot FOM with extreme solution for separated
    base.plot_snapshots(hash_fom_sep_ext)
    base.plot_initial(hash_fom_sep_ext)

    # Plot FOM with high solution for separated
    base.plot_snapshots(hash_fom_sep_hig)
    base.plot_snapshots(hash_fom_sep_med)
    base.plot_snapshots(hash_fom_sep_low)


    # Plot FOM with extreme solution for overlaped
    base.plot_snapshots(hash_fom_over_ext)
    base.plot_initial(hash_fom_over_ext)

    # Plot DECOMP with initialization
    base.plot_snapshots(hash_dec_init)
    base.plot_comparison(hash_dec_init,hash_fom_basis)
    base.plot_error(hash_dec_init)

    # Plot DECOMP with separated and shift and complete
    base.plot_snapshots(hash_dec_sep)
    base.plot_comparison(hash_dec_sep,hash_fom_sep_ext)

    # Plot DECOMP with separated no shift
    base.plot_snapshots(hash_dec_sep_ns)
    base.plot_states(hash_dec_sep_ns)
    base.plot_error(hash_dec_sep_ns)
    base.plot_comparison(hash_dec_sep_ns,hash_fom_sep_ext)

    # Plot DECOMP with separated 
    base.plot_snapshots(hash_dec_sep)
    base.plot_error(hash_dec_sep)
    base.plot_comparison(hash_dec_sep,hash_fom_sep_ext)

    # Plot Decomp with Simple
    base.plot_snapshots(hash_dec_sep_simp_1)
    base.plot_decomposition(hash_dec_sep_simp_1)
    base.plot_error(hash_dec_sep_simp_1)
    base.plot_snapshots(hash_dec_sep_simp_2)
    base.plot_decomposition(hash_dec_sep_simp_2)
    base.plot_error(hash_dec_sep_simp_2)
    base.plot_snapshots(hash_dec_sep_simp_3)
    base.plot_decomposition(hash_dec_sep_simp_3)
    base.plot_error(hash_dec_sep_simp_3)
    base.plot_snapshots(hash_dec_sep_simp_4)
    base.plot_decomposition(hash_dec_sep_simp_4)
    base.plot_error(hash_dec_sep_simp_4)

    # Plot Decomp with left and right wave
    base.plot_snapshots(hash_dec_sep_simp_left)
    base.plot_error(hash_dec_sep_simp_left)
    base.plot_snapshots(hash_dec_sep_simp_right)
    base.plot_error(hash_dec_sep_simp_right)
    base.plot_snapshots(hash_merged_sep)
    base.plot_error(hash_merged_sep)

    # Plot Decomp with Gauss
    base.plot_snapshots(hash_dec_sep_gaus_2)
    base.plot_comparison(hash_dec_sep_gaus_2,hash_fom_sep_ext)
    base.plot_error(hash_dec_sep_gaus_2)
    base.plot_decomposition(hash_dec_sep_gaus_2)

    # Plot DECOMP with overlapped no shift
    base.plot_snapshots(hash_dec_over_ns)
    base.plot_states(hash_dec_over_ns)
    base.plot_error(hash_dec_over_ns)
    base.plot_comparison(hash_dec_over_ns,hash_fom_over_ext)

    # Plot DECOMP with overlapped 
    base.plot_snapshots(hash_dec_over)
    base.plot_error(hash_dec_over)
    base.plot_comparison(hash_dec_over,hash_fom_over_ext)

    # Plot Decomp with Simple for overlapped
    base.plot_snapshots(hash_dec_over_simp_1)
    base.plot_decomposition(hash_dec_over_simp_1)
    base.plot_error(hash_dec_over_simp_1)
    base.plot_snapshots(hash_dec_over_simp_2)
    base.plot_decomposition(hash_dec_over_simp_2)
    base.plot_error(hash_dec_over_simp_2)
    base.plot_snapshots(hash_dec_over_simp_3)
    base.plot_decomposition(hash_dec_over_simp_3)
    base.plot_error(hash_dec_over_simp_3)
    base.plot_snapshots(hash_dec_over_simp_4)
    base.plot_decomposition(hash_dec_over_simp_4)
    base.plot_error(hash_dec_over_simp_4)

    # Plot Decomp with left and right wave for overlapped
    base.plot_snapshots(hash_dec_over_simp_left)
    base.plot_error(hash_dec_over_simp_left)
    base.plot_comparison(hash_dec_over_simp_left,hash_fom_over_ext)
    base.plot_snapshots(hash_dec_over_simp_right)
    base.plot_error(hash_dec_over_simp_right)
    base.plot_snapshots(hash_merged_over)
    base.plot_error(hash_merged_over)

    # Plot Decomp with Gauss for overlapped
    base.plot_snapshots(hash_dec_over_gaus_2)
    base.plot_comparison(hash_dec_over_gaus_2,hash_fom_over_ext)
    base.plot_error(hash_dec_over_gaus_2)
    base.plot_decomposition(hash_dec_over_gaus_2)

    # Plot ROM with separated
    base.plot_snapshots(hash_rom_sep_ext)
    base.plot_error(hash_rom_sep_ext)
    base.plot_states(hash_rom_sep_ext)
    base.plot_comparison(hash_rom_sep_ext,hash_fom_basis)

    base.plot_snapshots(hash_rom_sep_low)
    base.plot_error(hash_rom_sep_low)
    base.plot_states(hash_rom_sep_low)
    base.plot_comparison(hash_rom_sep_low,hash_fom_basis)

    # Plot ROM with overlaped
    base.plot_snapshots(hash_rom_over_ext)
    base.plot_error(hash_rom_over_ext)
    base.plot_comparison(hash_rom_over_ext,hash_fom_basis)
    base.plot_states(hash_rom_over_ext)

    base.plot_snapshots(hash_rom_over_low)
    base.plot_error(hash_rom_over_low)
    base.plot_comparison(hash_rom_over_low,hash_fom_basis)
    base.plot_states(hash_rom_over_low)
    
    # Deactivate all plot commands are activated
    plt.show()