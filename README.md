# PyRICE2020
Stochastic simulation model implementation of the RICE2010 model by Nordhaus

## Goal of the PyRICE model
The PyRICE model can be used to generate emission abatement pathways in the RICE model that are optimzed over multiple alternative distributive objectives. Alternative distributive objectives have been added to the RICE model contrary to the Utilitarian objective in the original RICE 2010 Model. In essence, the PyRICE model generates abatement pathways that are deemed 'optimal' if the social planner would emphasize other goals than just maximizing aggregated welfare. 

## Uncertainty modules
The PyRICE has been connected to the SSP scenarios by aggregating country statistics into the RICE regions. Extra climate uncertainties have been added to analyse expsosure of alternative abatement pathways to deep uncertainty in climate change. Within the model, it can be switched between a long term and short term uncertainty analysis. 

To use the uncertainty modules, additional packages need to be installed to connect to the EMA-workbench: 
https://emaworkbench.readthedocs.io/en/latest/

## Availabtility of data
Due to the large file size of the generated results, these are not available in this repository. These data files can be provided on request. 
