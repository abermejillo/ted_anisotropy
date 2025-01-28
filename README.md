# ted_anisotropy
Theory work on describing the dissipation in magnetic van der Waals nanomechanical resonators across the phase transition.

In data we store some of the calculations of the thermodynamic properties and data obtained from other papers.

thermal_property_calculator.py contains functions used to compute the thermodynamic properties of a 2D magnet

thermal_properties_FEPS.ipynb shows how we use the functions in thermal_property_calculator.py to compute the thermodynamic properties of FePS3 (figures 3, 4)

dissipation.py contains functions to compute the dissipated energy as a function of device and material parameters

strain_heat_distribution.ipynb shows code that computes the temperature and strain profiles in a bent plate as shown in the manuscript (figure 2)

anisotropic_dissipation_analysis.ipynb shows how we use the functions of dissipation.py to obtain the results shown in the manuscript (figure 5, 6, 8)

ratios_analysis.ipynb shows how we use the functions of dissipation.py to obtain results for different geometries and ratios of k_inplane vs k_outofplane, the last result is included in the publication (figure 7)


