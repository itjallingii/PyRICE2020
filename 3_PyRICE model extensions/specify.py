from ema_workbench import Scenario

def specify_levers(sr = 0, irstp = 0, periodfullpart = 0, miu_period = 0) :
    
    '''Specify levers for further usage'''
    
    levers_param = {'sr': sr, 'irstp': irstp, 'periodfullpart': periodfullpart, 
                  'miu_period': miu_period}
    
    return levers_param

def specify_scenario(reference_values, dice_sm) :
    scen = {}
    for key in dice_sm.uncertainties:
        scen.update({key.name: reference_values[key.name]})
    reference_scenario = Scenario('reference', **scen)
    return reference_scenario

# def default_scenario(dike_model) : 
#     reference_values = {'Bmax': 175, 'Brate': 1.5, 'pfail': 0.5,
#                         'discount rate': 3.5,
#                         'ID flood wave shape': 4}
#     scen1 = {}

#     for key in dike_model.uncertainties:
#         name_split = key.name.split('_')

#         if len(name_split) == 1:
#             scen1.update({key.name: reference_values[key.name]})

#         else:
#             scen1.update({key.name: reference_values[name_split[1]]})

#     ref_scenario = Scenario('reference', **scen1)
#     return ref_scenario
