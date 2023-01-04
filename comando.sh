#!/bin/bash
#commented to be used with docker, uncommented to be used by command line in a local env.
#PYTHONPATH=/ariel/DataScience/Gusano/BiMPuRLE
#export PYTHONPATH
specArch="NO"
if [ "$BIM_COMM" == "optimize" ]
then
configVar=$BIM_MOD$BIM_OPT$BIM_STE$BIM_BAT$BIM_WOR
fi
if [ "$BIM_COMM" == "replay" ]
then
echo "replay comun"
configLab=$BIM_LAB
configVar=$BIM_MOD$BIM_OPT$BIM_STE$BIM_BAT$BIM_WOR$BIM_FOL
echo "confVar:$configVar"
echo "comando:$BIM_COMM"
echo "folder: $BIM_FOL"
fi

if [ "$BIM_COMM" == "replaySA" ]
then
specArch="SA"	
configLab=$BIM_LAB
configVar=$BIM_MOD$BIM_OPT$BIM_STE$BIM_BAT$BIM_WOR$BIM_FOL
BIM_COMM="replay"
export BIM_COMM
BIM_FOL="shuffleArch/$BIM_FOL"
export BIM_FOL
echo "confVar:$configVar"
echo "comando:$BIM_COMM"
echo "folder:$BIM_FOL"
fi

if [ "$BIM_COMM" == "modelPulseStats" ]
then
configVar=$BIM_MOD$BIM_STE
fi
if [ "$BIM_COMM" == "shuffleArch" ]
then
configVar=$BIM_MOD$BIM_OPT$BIM_STE$BIM_BAT$BIM_WOR
fi

time(python ./RLEngine/RunFromEnv.py)&>log_$configVar.log
echo "$(cat log_$configVar.log)"

userTime="$(grep user log_$configVar.log|awk {'print $2'})"
sysTime="$(grep sys log_$configVar.log|awk {'print $2'})"
realTime="$(grep real log_$configVar.log|awk {'print $2'})"

if [ "$BIM_COMM" == "optimize" ]
then
echo "Ran an optimize"
folder="$(grep Model log_$configVar.log|awk -F "/" {'print $2"/"$3'})"
echo "$configVar|$userTime|$sysTime|$realTime|$folder" >> ./uid.0/$BIM_MOD"_"$BIM_ENV"_"$BIM_OPT/trainigTimeResults$BIM_MOD$BIM_OPT$BIM_STE
fi

if [ "$BIM_COMM" == "shuffleArch" ]
then
echo "Ran a shuffleArch optimize"
folder="$(grep Model log_$configVar.log|awk -F "/" {'print $2"/"$3"/"$4'})"
echo "$configVar|$userTime|$sysTime|$realTime|$folder" >> ./uid.0/$BIM_MOD"_"$BIM_ENV"_"$BIM_OPT/shuffleArch/trainigTimeResults$BIM_MOD$BIM_OPT$BIM_STE
fi


if [ "$BIM_COMM" == "replay" ] && [ "$specArch" == "NO" ]
then
echo "Ran specArch replay: $specArch"	
folder="$(grep 'replay con rootpath' log_$configVar.log|awk -F "/" {'print $2"/"$3'})"
echo "$configLab|$userTime|$sysTime|$realTime|$folder" >> ./uid.0/$BIM_MOD"_"$BIM_ENV"_"$BIM_OPT/replayTimeResults$BIM_MOD$BIM_OPT$BIM_STE
fi


if [ "$BIM_COMM" == "replay" ] && [ "$specArch" == "SA" ]
then
echo "Ran specArch replay: $specArch"  
folder="$(grep 'replay con rootpath' log_$configVar.log|awk -F "/" {'print $2"/"$3"/"$4'})"
echo "$configLab|$userTime|$sysTime|$realTime|$folder" >> ./uid.0/$BIM_MOD"_"$BIM_ENV"_"$BIM_OPT/replayTimeResults$BIM_MOD$BIM_OPT$BIM_STE
fi





if [ "$BIM_COMM" == "modelPulseStats" ]
then
echo "Ran a modelPulseStats"  
echo "$configVar|$userTime|$sysTime|$realTime" >> ./uid.0/pulseTimeResults
fi



rm -rf ./log_$configVar.log

