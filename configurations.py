
## Configurations ##
####    these variables need to be edited if running from different folders/PCs    ####
main_folder = r"C:\Users\python_student\fabian\data_lab_report\exp2"  ## main folder where the different subfolders are in
group1 = main_folder+r"\high_pH" ## folder group 1, keep \ that way
group2 = main_folder+r"\high_pH_cysteine" ## folder group 2, keep \ that way
group3 = main_folder+r"\normal_pH_cysteine" ## folder group 3 keep \ that way
group4 = main_folder+r"\pre\pre_high_pH" ## folder group 1, keep \ that way
group5 = main_folder+r"\pre\pre_high_pH_cysteine" ## folder group 2, keep \ that way
group6 = main_folder+r"\pre\pre_normal_pH_cysteine" ## folder group 3 keep \ that way

group_number = 6 # prevents issue with for variable in locals()

cascade_file_path = r"C:\Users\python_student\Cascade-master" ## CASCADE master folder

frame_rate = 10

## plot a set of nb_neurons randomly chosen neuronal traces (first seconds)
nb_neurons = 16 ## maybe put directly into cascade_this???

model_name = "Global_EXC_10Hz_smoothing200ms" 
## select fitting model from list (created in cascada code) ##
## list still in CASCADE code, maybe add here##

EXPERIMENT_DURATION = 60
 
FRAME_INTERVAL = 1 / frame_rate
 
BIN_WIDTH =  20 
#SET TO APPROX 200ms
 
FILTER_NEURONS = True

groups = []
for n in range(group_number):
    group_name = f"group{n+1}"
    if group_name in locals():
        groups.append(locals()[group_name])

