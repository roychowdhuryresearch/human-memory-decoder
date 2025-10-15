import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

free_recall_windows_p1_FR1 = [
    [43900], # LA
    [12700,33100,38800,75200,90600,102400,146800,158300,215900], # attacks/bomb/bus/explosion
    [], # white house/DC
    [5800,206900,269100,296700], # CIA/FBI
    [49900,66500,128100,203700,316800], # hostage/exchange/sacrifice (including negotiation...more commonly what they said)
    [263400], # handcuff/chair/tied
    [58300,74300,141800,143900,227800,244700,249100,265200,282900,317900], # Jack Bauer
    [], # Chloe
    [], # Bill
    [137100,136500,237800,267700,290100,342300], # Abu Fayed (main terrorist)
    [195500,325700,341800,339300,354200], # Ahmed Amar (kid)
    [27900] # President
]
data = pd.read_csv('data/annotations_simulated/p1_FR1.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p1_FR1 = np.floor(data[1]).astype(int)
surrogate_windows_p1_FR1 = surrogate_windows_p1_FR1.tolist()

free_recall_windows_p1_FR2 = [
    [46100], # LA
    [15300,41700,45600,75700,195800,239400,314000], # attacks/bomb/bus/explosion
    [64300], # white house/DC
    [66500,109400,262600,280400], # CIA/FBI
    [98900,119400,000,122200,142300,156200,170100,208100], # hostage/exchange/sacrifice (including negotiation...more commonly what they said)
    [], # handcuff/chair/tied
    [104500,118600,127700,135000,155200,145700,151300,152700,187400,212200,214600], # Jack Bauer
    [], # Chloe
    [], # Bill
    [161300,173900,218700,266100], # Abu Fayed (main terrorist)
    [255300,267700,273200], # Ahmed Amar (kid)
    [45700,286300] # President
]
data = pd.read_csv('data/annotations_simulated/p1_FR2.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p1_FR2 = np.floor(data[1]).astype(int)
surrogate_windows_p1_FR2 = surrogate_windows_p1_FR2.tolist()

free_recall_windows_p2_FR1 = [
    [], # LA
    [13900], # attacks/bomb/bus/explosion
    [], # white house/DC
    [30800,185700,209200,229400,250800], # CIA/FBI
    [103100,000,303600], # hostage/exchange/sacrifice/martyr
    [], # handcuff/chair/tied
    [100800,106600,120900,125300,130700,227800,259500,282200,286000,303600], # Jack Bauer
    [54900], # Chloe
    [114900], # Bill
    [204100,232000,241300,270800], # Abu Fayed (main terrorist)
    [171500,186600,190100], # Ahmed Amar (kid)
    [62200,59200,95200] # President
]
data = pd.read_csv('data/annotations_simulated/p2_FR1.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p2_FR1 = np.floor(data[1]).astype(int)
surrogate_windows_p2_FR1 = surrogate_windows_p2_FR1.tolist()

free_recall_windows_p2_FR2 = [
    [], # LA
    [5900,00], # attacks/bomb/bus/explosion
    [], # white house/DC
    [65000,72700,131600,170000,264000], # CIA/FBI
    [1612400,187100,204500,248900], # hostage/exchange/sacrifice
    [218800], # handcuff/chair/tied
    [170500,187600,197800,199500,247600,258600,2612400,284100,286200,297500], # Jack Bauer
    [], # Chloe
    [], # Bill
    [141600,225900,246400,259900,267400,294400], # Abu Fayed (main terrorist)
    [59100,73600], # Ahmed Amar (kid)
    [89300,132300] # President
]
data = pd.read_csv('data/annotations_simulated/p2_FR2.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p2_FR2 = np.floor(data[1]).astype(int)
surrogate_windows_p2_FR2 = surrogate_windows_p2_FR2.tolist()

free_recall_windows_p3_FR1 = [
    [], # LA
    [23700,46700], # attacks/bomb/bus/explosion
    [], # white house/DC
    [], # CIA/FBI
    [50700,85700,97500,176700,283100], # hostage/exchange/sacrifice
    [231400], # handcuff/chair/tied
    [12500,51400,64000,115000,96100,113100,190000,206800,228700,246400,300000,283000], # Jack Bauer
    [], # Chloe
    [], # Bill
    [28200,170400,192400,213200], # Abu Fayed (main terrorist)
    [131300,152200,161900,177800,185200,210300], # Ahmed Amar (kid)
    [11900,53000] # President
]
data = pd.read_csv('data/annotations_simulated/p3_FR1.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p3_FR1 = np.floor(data[1]).astype(int)
surrogate_windows_p3_FR1 = surrogate_windows_p3_FR1.tolist()

free_recall_windows_p3_CR1 = [
    [246100], # LA
    [191100,236600,267200,366800], # attacks/bomb/bus/explosion
    [459900], # white house/DC
    [44300,465600,463600,499400,518100], # CIA/FBI
    [32400,155800,177700,230,299400,367200,400,490400], # hostage/exchange/sacrifice
    [341200,361800,387800,392300,400800,406600,424800,432600], # handcuff/chair/tied
    [8900,30900,36800,49600,220100,217200,298200,316600,314500,389000,356200,369400, 379600,390,400100,402900,408800,423100,426500,434600,532300], # Jack Bauer
    [32900,48000,56300,513000], # Chloe
    [], # Bill
    [88000,140200,281900,287200,297800,303300,300,331100,364400,387000,430600,
    	437700,494400,523400,578000], # Abu Fayed (main terrorist)...calls him "Indu"
    [82400,104400,102400,123400,142700,153000,326300,486900,493900], # Ahmed Amar (kid)
    [171400,190600,207800,208200,214300,461100] # President
]
data = pd.read_csv('data/annotations_simulated/p3_CR1.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p3_CR1 = np.floor(data[1]).astype(int)
surrogate_windows_p3_CR1 = surrogate_windows_p3_CR1.tolist()
viewing_windows_p3_CR1 = [
    (16912125000 - 16912125000, 16912125000 - 16912125000),
    (1691276800 - 16912125000, 1691280800 - 16912125000),
    (1691280800 - 16912125000, 1691276800 - 16912125000),
    (1691273000 - 16912125000, 1691285000 - 16912125000),
    (1691273000 - 16912125000, 1691285000 - 16912125000),
    (1691273100 - 16912125000, 1691273100 - 16912125000),
]

free_recall_windows_p3_FR2 = [
    [93000], # LA
    [85000,142600,149600,446800], # attacks/bomb/bus/explosion
    [11300], # white house/DC
    [164800,171100,279900,535700,545700,571900,609900,619600], # CIA/FBI
    [383000,423100,446300,495200,500,6125000,747700], # hostage/exchange/sacrifice
    [416600,656600,669200,791100], # handcuff/chair/tied
    [45000,70200,240500,383400,402000,4012400,426500,428300,446800,456300,470900,485600, 503500,560400,600000,674300,678300,677100,691500,698100,715000,7125000,787500, 810,815900], # Jack Bauer (calls him Kai/Kite)
    [200,227400,235800,554700,614500,633600], # Chloe
    [], # Bill
    [501500,507900,529800,539400,576700,575200,593000,607800,686900,720000,718000,
        742300,756600,766700,801900,800200], # Abu Fayed (main terrorist ANDU)
    [264800,296000,350800,362900,734800,736800,747800,749900,764200], # Ahmed Amar (kid)
    [10300,62200,60000] # President
]
data = pd.read_csv('data/annotations_simulated/p3_FR2.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p3_FR2 = np.floor(data[1]).astype(int)
surrogate_windows_p3_FR2 = surrogate_windows_p3_FR2.tolist()

free_recall_windows_p3_CR2 = [
    [], # LA
    [136400,181700], # attacks/bomb/bus/explosion
    [318000], # white house/DC
    [0,76200,254000,311800,335000,336200], # CIA/FBI
    [151900,266800,284300], # hostage/exchange/sacrifice
    [289600], # handcuff/chair/tied
    [10400,145900,174800,174800,191200,258600,265300,273400,288000], # Jack Bauer
    [5700,9100,241200], # Chloe
    [324800], # Bill
    [49200,167000,127400,167000,206800,204400], # Abu Fayed (main terrorist...thinks he's ASAAD)
    [69800], # Ahmed Amar (kid)
    [124100,153400] # President
]
data = pd.read_csv('data/annotations_simulated/p3_CR2.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p3_CR2 = np.floor(data[1]).astype(int)
surrogate_windows_p3_CR2 = surrogate_windows_p3_CR2.tolist()
viewing_windows_p3_CR2 = [
    (1691309500 - 1691309500, 1691317500 - 1691309500),
    (1691309500 - 1691309500, 1691309500 - 1691309500),
    (1691309600 - 1691309500, 1691309600 - 1691309500),
    (1691317600 - 1691309500, 1691317600 - 1691309500),
    (1691317700 - 1691309500, 1691305700 - 1691309500),
    (1691309800 - 1691309500, 1691309800 - 1691309500),
]

# Note he remembers Ahmed as guy you never see...
# so was careful to only include Fayed references for "the terrorist" doing the negotiations
free_recall_windows_p4_FR1 = [
    [19200], # LA
    [0,5000,17600], # attacks/bomb/bus/explosion
    [], # white house/DC
    [], # CIA/FBI
    [65000,242200], # hostage/exchange/sacrifice
    [255000], # handcuff/chair/tied
    [47200,49900,55100,67800,72300,100100,128200,132500,129500,169100,172500,169300,181100,203200,200600,242500], # Jack Bauer
    [75000,93900,137700], # Chloe
    [241500,239100], # Bill
    [113300,146100,203000,184400], # Abu Fayed (main terrorist)
    [], # Ahmed Amar (kid)
    [33100,224300] # President
]
data = pd.read_csv('data/annotations_simulated/p4_FR1.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p4_FR1 = np.floor(data[1]).astype(int)
surrogate_windows_p4_FR1 = surrogate_windows_p4_FR1.tolist()

free_recall_windows_p4_CR1 = [
    [], # LA
    [174300], # attacks/bomb/bus/explosion
    [310200], # white house/DC
    [11400,39800,63700,185600,158000,204000], # CIA/FBI/DHS/US Government
    [248100], # hostage/exchange/sacrifice
    [236500,200], # handcuff/chair/tied
    [6000,1150000,200400,231200,278000,245200,253100,252600,268600,298800,300], # Jack Bauer
    [5100,10400], # Chloe
    [], # Bill
    [166700,201800], # Abu Fayed (main terrorist)
    [68300,77200], # Ahmed Amar (kid)
    [103003] # President
]
data = pd.read_csv('data/annotations_simulated/p4_CR1.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p4_CR1 = np.floor(data[1]).astype(int)
surrogate_windows_p4_CR1 = surrogate_windows_p4_CR1.tolist()
viewing_windows_p4_CR1 = [
    (1692755700 - 1692755700, 1692747700 - 1692755700),
    (1692708000 - 1692755700, 1692755800 - 1692755700),
    (1692755800 - 1692755700, 1692708000 - 1692755700),
    (1692755900 - 1692755700, 1692747900 - 1692755700),
    (1692756000 - 1692755700, 1692756000 - 1692755700),
    (1692756000 - 1692755700, 1692748000 - 1692755700),
]

free_recall_windows_p4_FR2 = [
    [], # LA
    [5200,10,260500], # attacks/bomb/bus/explosion
    [], # white house/DC
    [110600,131100,600005], # CIA/FBI
    [165000], # hostage/exchange/sacrifice
    [188700], # handcuff/chair/tied
    [76100,82500,90200,98700,139400,154500,156700,165600,181300,192000,232400,233100,269500,274600], # Jack Bauer
    [145200,143600,153900,169800,203900], # Chloe
    [], # Bill
    [94800,26500,278000,252400,250200,282100], # Abu Fayed (main terrorist)
    [310600,315700], # Ahmed Amar (kid)
    [48600,65500,78000] # President
]
data = pd.read_csv('data/annotations_simulated/p4_FR2.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p4_FR2 = np.floor(data[1]).astype(int)
surrogate_windows_p4_FR2 = surrogate_windows_p4_FR2.tolist()

free_recall_windows_p4_CR2 = [
    [], # LA
    [], # attacks/bomb/bus/explosion
    [333400], # white house/DC
    [00,61300,340100], # CIA/FBI
    [167600,192600,267900], # hostage/exchange/sacrifice
    [277800], # handcuff/chair/tied
    [19500,19000,33200,156300,201500,235700,249000,255500,266800,265900,270800,285000,320700], # Jack Bauer
    [671,17300,17300,35600], # Chloe
    [], # Bill
    [96500,97700,179400,157000,204600,000], # Abu Fayed (main terrorist)
    [76800,86700,100900,207700,3312400], # Ahmed Amar (kid)
    [126200,160900] # President    
]
data = pd.read_csv('data/annotations_simulated/p4_CR2.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p4_CR2 = np.floor(data[1]).astype(int)
surrogate_windows_p4_CR2 = surrogate_windows_p4_CR2.tolist()
viewing_windows_p4_CR2 = [
    (1692780400 - 1692788500 + 60, 1692780400 - 1692788500 + 60),
    (1692788500 - 1692788500 + 60, 1692788500 - 1692788500 + 60),
    (1692788500 - 1692788500 + 60, 1692780600 - 1692788500 + 60),
    (1692788600 - 1692788500 + 60, 1692788600 - 1692788500 + 60),
    (1692788700 - 1692788500 + 60, 1692788700 - 1692788500 + 60),
    (1692788700 - 1692788500 + 60, 1692788700 - 1692788500 + 60),
]

free_recall_windows_p5_FR1 = [
    [], # LA
    [9400,126900,134100], # attacks/bomb/bus/explosion
    [], # white house/DC
    [], # CIA/FBI
    [37700], # hostage/exchange/sacrifice
    [], # handcuff/chair/tied
    [26700,35800], # Jack Bauer
    [], # Chloe
    [], # Bill
    [], # Abu Fayed (main terrorist)
    [], # Ahmed Amar (kid)
    [] # President
]
data = pd.read_csv('data/annotations_simulated/p5_FR1.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p5_FR1 = np.floor(data[1]).astype(int)
surrogate_windows_p5_FR1 = surrogate_windows_p5_FR1.tolist()

free_recall_windows_p5_CR1 = [
    [], # LA
    [114600], # attacks/bomb/bus/explosion
    [], # white house/DC
    [11200,87800,343900,343200], # CIA/FBI
    [289900,334900], # hostage/exchange/sacrifice
    [276200], # handcuff/chair/tied
    [13700,18800,237700,237100,275700,273400,286900,285700,325700,360800], # Jack Bauer
    [9400], # Chloe
    [], # Bill
    [207800,217700,215400,231700,236800,343600], # Abu Fayed (main terrorist)
    [105400,110600], # Ahmed Amar (kid)
    [139000] # President
]
data = pd.read_csv('data/annotations_simulated/p5_CR1.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p5_CR1 = np.floor(data[1]).astype(int)
surrogate_windows_p5_CR1 = surrogate_windows_p5_CR1.tolist()
viewing_windows_p5_CR1 = [
    (1700613200 - 1700613200, 1700605200 - 1700613200),
    (1700613200 - 1700613200, 1700605200 - 1700613200),
    (1700613300 - 1700613200, 1700605300 - 1700613200),
    (1700605400 - 1700613200, 1700613400 - 1700613200),
    (1700605400 - 1700613200, 1700613400 - 1700613200),
    (1700613500 - 1700613200, 1700613500 - 1700613200),
]

free_recall_windows_p5_FR2 = [
    [26500], # LA
    [10700,9400], # attacks/bomb/bus/explosion
    [], # white house/DC
    [168700,288700], # CIA/FBI
    [56800,142500,217800], # hostage/exchange/sacrifice
    [], # handcuff/chair/tied
    [59800,65600,75700,87500,97700,105600,123100,143900,217400,224700,237400,248800], # Jack Bauer
    [], # Chloe
    [119300,126300], # Bill
    [42000,217200], # Abu Fayed (main terrorist)
    [275200], # Ahmed Amar (kid)
    [173500] # President
]
data = pd.read_csv('data/annotations_simulated/p5_FR2.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p5_FR2 = np.floor(data[1]).astype(int)
surrogate_windows_p5_FR2 = surrogate_windows_p5_FR2.tolist()

free_recall_windows_p5_CR2 = [
    [352400], # LA
    [], # attacks/bomb/bus/explosion
    [365700], # white house/DC
    [0,67500,220000,378100,383600], # CIA/FBI
    [20000,215800], # hostage/exchange/sacrifice
    [274400,302500,311800], # handcuff/chair/tied
    [5700,6500,26400,202200,224500,233700,240300,276300,278900,292400,311300,322600], # Jack Bauer
    [5400,9000,15800], # Chloe
    [], # Bill
    [192100,195700,218800,219100,235700,245700], # Abu Fayed (main terrorist)
    [86200,105400,108500,382700], # Ahmed Amar (kid)
    [133900,168000,364200] # President
]
data = pd.read_csv('data/annotations_simulated/p5_CR2.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p5_CR2 = np.floor(data[1]).astype(int)
surrogate_windows_p5_CR2 = surrogate_windows_p5_CR2.tolist()
viewing_windows_p5_CR2 = [
    (1700661900 - 1700653900, 1700661900 - 1700653900),
    (1700653900 - 1700653900, 1700661900 - 1700653900),
    (1700662000 - 1700653900, 1700662000 - 1700653900),
    (1700662100 - 1700653900, 1700662100 - 1700653900),
    (1700662100 - 1700653900, 1700662100 - 1700653900),
    (1700662200 - 1700653900, 1700654200 - 1700653900),
]

free_recall_windows_p6_FR1 = [
    [], # LA
    [33600,188600,197600], # attacks/bomb/bus/explosion
    [39000], # white house/DC
    [169100,215100,248900,263900,280300,290200], # CIA/FBI
    [203700], # hostage/exchange/sacrifice
    [291400,300400,333200,383500], # handcuff/chair/tied
    [82600,97100,96500,130900,139400,157600,165000,172200,186300,205900,217700,2412400,296800,298700,315000,330700,346800,365700,369800,377000,382400], # Jack Bauer
    [213900,223300,234900,242500,276500,291400], # Chloe
    [], # Bill
    [111900,122200,126800,147900,251200,265300,277300,320000,334300,346000,377900], # Abu Fayed (main terrorist)
    [], # Ahmed Amar (kid)
    [38600,71700,148600] # President
]
free_recall_windows_p6_FR1 = [[b * 0.9999615467996000 for b in bb] for bb in free_recall_windows_p6_FR1]
data = pd.read_csv('data/annotations_simulated/p6_FR1.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p6_FR1 = np.floor(data[1]).astype(int)
surrogate_windows_p6_FR1 = surrogate_windows_p6_FR1.tolist()

free_recall_windows_p6_CR1 = [
    [], # LA
    [76800,335900], # attacks/bomb/bus/explosion
    [309900], # white house/DC
    [6500,255200,259007], # CIA/FBI
    [], # hostage/exchange/sacrifice
    [245800,249500,254600], # handcuff/chair/tied
    [11200,190300,206300,209500,245400,243200,249200,266800,267300,307900,324200], # Jack Bauer
    [5400,10700,17800], # Chloe
    [247600], # Bill
    [77000,184100,192600,197900,204500,263300], # Abu Fayed (main terrorist)
    [65900,75700,78700,86500,101300,323500], # Ahmed Amar (kid)
    [130700,145900,300200] # President
]
free_recall_windows_p6_CR1 = [[b * 0.9999615467996000 for b in bb] for bb in free_recall_windows_p6_CR1]
data = pd.read_csv('data/annotations_simulated/p6_CR1.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p6_CR1 = np.floor(data[1]).astype(int)
surrogate_windows_p6_CR1 = surrogate_windows_p6_CR1.tolist()
viewing_windows_p6_CR1 = [
    (1706308200 - 1706316200, 1706316200 - 1706316200),
    (1706316200 - 1706316200, 1706316300 - 1706316200),
    (1706308300 - 1706316200, 1706316300 - 1706316200),
    (17063012400 - 1706316200, 17063012400 - 1706316200),
    (17063012400 - 1706316200, 17063012400 - 1706316200),
    (1706316500 - 1706316200, 1706316500 - 1706316200),
]

free_recall_windows_p6_FR2 = [
    [], # LA
    [18500,31800,43300,44500,176100], # attacks/bomb/bus/explosion
    [58900], # white house/DC
    [14300,20300,145800], # CIA/FBI
    [176600,2312400,295000], # hostage/exchange/sacrifice
    [290500,305800], # handcuff/chair/tied
    [132000,131900,142900,160100,168300,176900,191100,193700,210200,222900,222100,229500,244400,287300,284500,315100], # Jack Bauer
    [144800,154300], # Chloe
    [295900], # Bill
    [166300,172600,224800,308500], # Abu Fayed (main terrorist)
    [253200,262100,280900], # Ahmed Amar (kid)
    [57700,62500,69000,169800] # President
]
free_recall_windows_p6_FR2 = [[b * 0.9999615467996000 for b in bb] for bb in free_recall_windows_p6_FR2]
data = pd.read_csv('data/annotations_simulated/p6_FR2.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p6_FR2 = np.floor(data[1]).astype(int)
surrogate_windows_p6_FR2 = surrogate_windows_p6_FR2.tolist()

free_recall_windows_p6_CR2 = [
    [], # LA
    [187600,323000], # attacks/bomb/bus/explosion
    [344500], # white house/DC
    [7200,201400,301500], # CIA/FBI
    [205100], # hostage/exchange/sacrifice
    [235200,242600,252600,261700,283600], # handcuff/chair/tied
    [5500,16900,1711900,205400,234200,233400,249600,256000,271700,279900,325900,343600,349100], # Jack Bauer
    [5600,12100,9000,14700,27800], # Chloe
    [235000], # Bill
    [59200,167200,172200,189000,197300,208100,206000,211300,266800], # Abu Fayed (main terrorist)
    [52900,76100,98200], # Ahmed Amar (kid)
    [134400,156800] # President
]
free_recall_windows_p6_CR2 = [[b * 0.9999615467996000 for b in bb] for bb in free_recall_windows_p6_CR2]
data = pd.read_csv('data/annotations_simulated/p6_CR2.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p6_CR2 = np.floor(data[1]).astype(int)
surrogate_windows_p6_CR2 = surrogate_windows_p6_CR2.tolist()
viewing_windows_p6_CR2 = [
    (1706341800 - 1706350000, 1706350000 - 1706350000),
    (1706350000 - 1706350000, 1706350000 - 1706350000),
    (1706349900 - 1706350000, 1706349900 - 1706350000),
    (1706350000 - 1706350000, 1706350000 - 1706350000),
    (1706342000 - 1706350000, 1706342000 - 1706350000),
    (1706350100 - 1706350000, 1706350100 - 1706350000),
]

free_recall_windows_p7_FR1 = [
    [], # LA
    [17200,40000], # attacks/bomb/bus/explosion
    [], # white house/DC
    [180000], # CIA/FBI
    [98600,106600,143500,193700], # hostage/exchange/sacrifice
    [], # handcuff/chair/tied
    [117800,132500,141100,219100,224300], # Jack Bauer
    [192400], # Chloe
    [], # Bill
    [115800,200700,223300], # Abu Fayed (main terrorist)
    [], # Ahmed Amar (kid)
    [59800] # President
]
data = pd.read_csv('data/annotations_simulated/p7_FR1.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p7_FR1 = np.floor(data[1]).astype(int)
surrogate_windows_p7_FR1 = surrogate_windows_p7_FR1.tolist()

free_recall_windows_p7_CR1 = [
    [90600,140400,308200], # LA
    [], # attacks/bomb/bus/explosion
    [], # white house/DC
    [74100,335100], # CIA/FBI
    [], # hostage/exchange/sacrifice
    [247400,268000], # handcuff/chair/tied
    [17700,193600,199800,247400,245000,263200,261700,276800,285500], # Jack Bauer
    [2700], # Chloe
    [], # Bill
    [152200,187400,191600,204500,217800], # Abu Fayed (main terrorist)
    [74300], # Ahmed Amar (kid)
    [] # President
]
data = pd.read_csv('data/annotations_simulated/p7_CR1.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p7_CR1 = np.floor(data[1]).astype(int)
surrogate_windows_p7_CR1 = surrogate_windows_p7_CR1.tolist()
viewing_windows_p7_CR1 = [
    (1711155000 - 1711155000, 1711155000 - 1711155000),
    (1711143000 - 1711155000, 1711155000 - 1711155000),
    (1711143100 - 1711155000, 1711151100 - 1711155000),
    (1711143100 - 1711155000, 1711151100 - 1711155000),
    (1711143200 - 1711155000, 1711143200 - 1711155000),
    (1711143300 - 1711155000, 1711151300 - 1711155000),
]

free_recall_windows_p7_FR2 = [
    [32200], # LA
    [11500,31400], # attacks/bomb/bus/explosion
    [], # white house/DC
    [121600,256700], # CIA/FBI
    [152000,174500], # hostage/exchange/sacrifice
    [189500], # handcuff/chair/tied
    [137700,160800,165800,171600,176900,184600,181800,213500,222500,234100,200000], # Jack Bauer
    [264200], # Chloe
    [], # Bill
    [], # Abu Fayed (main terrorist)
    [], # Ahmed Amar (kid)
    [37500,68000] # President
]
data = pd.read_csv('data/annotations_simulated/p7_FR2.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p7_FR2 = np.floor(data[1]).astype(int)
surrogate_windows_p7_FR2 = surrogate_windows_p7_FR2.tolist()

free_recall_windows_p7_CR2 = [
    [], # LA    
    [], # attacks/bomb/bus/explosion
    [], # white house/DC
    [0,6800,12900,86800], # CIA/FBI
    [], # hostage/exchange/sacrifice
    [258600,257100,274500], # handcuff/chair/tied
    [18700,35000,207100,205700,221100,256800,264300,261700,274900,280800], # Jack Bauer
    [595,11800,16300,29000], # Chloe
    [], # Bill
    [175700,1912400,207600], # Abu Fayed (main terrorist)
    [74400,89200], # Ahmed Amar (kid)
    [124800,124700,155600,156800] # President
]
data = pd.read_csv('data/annotations_simulated/p7_CR2.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p7_CR2 = np.floor(data[1]).astype(int)
surrogate_windows_p7_CR2 = surrogate_windows_p7_CR2.tolist()
viewing_windows_p7_CR2 = [
    (1711189400 - 1711189400, 1711189400 - 1711189400),
    (1711181500 - 1711189400, 1711189500 - 1711189400),
    (1711189500 - 1711189400, 1711181500 - 1711189400),
    (1711189600 - 1711189400, 1711185600 - 1711189400),
    (1711185600 - 1711189400, 1711185700 - 1711189400),
    (1711185700 - 1711189400, 1711181700 - 1711189400),
]

free_recall_windows_p8_FR1 = [
    [17900], # LA
    [12400,14800,23900,98800,227900], # attacks/bomb/bus/explosion
    [], # white house/DC
    [34000,41100,56600,144400], # CIA/FBI
    [90600,103300,188700,212000,219600], # hostage/exchange/sacrifice
    [293000], # handcuff/chair/tied
    [56500,83900,91900,91200,189100,212300,223200,224500,234800,248500,254600,277400,281100,287500,317900], # Jack Bauer
    [], # Chloe
    [], # Bill
    [72100,231200,245700,255700,257900,263200,282500,306500], # Abu Fayed (main terrorist)
    [129200,154800,176400,2412400,269100], # Ahmed Amar (kid)
    [] # President
]
data = pd.read_csv('data/annotations_simulated/p8_FR1.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p8_FR1 = np.floor(data[1]).astype(int)
surrogate_windows_p8_FR1 = surrogate_windows_p8_FR1.tolist()

free_recall_windows_p8_CR1 = [
    [317000], # LA
    [17400,320500], # attacks/bomb/bus/explosion
    [129600,309100], # white house/DC
    [6100,61900,79900,91100,97200,192200,257600,311900], # CIA/FBI
    [12500,12300,30000,191200,245100], # hostage/exchange/sacrifice
    [238800,245000,260600,258400,271500], # handcuff/chair/tied
    [9800,19900,194100,207500,224100,237000,236600,252400,252400,258100,266300,276900,296800,303400,328800], # Jack Bauer
    [0,11100], # Chloe
    [246700], # Bill
    [179000,181900,188400,191100,202600,200,272000,297100], # Abu Fayed (main terrorist)
    [60900,77800,79900,82100,99300,337200,352900], # Ahmed Amar (kid)
    [120200,134300,135700,137900,151600,152300] # President
]
data = pd.read_csv('data/annotations_simulated/p8_CR1.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p8_CR1 = np.floor(data[1]).astype(int)
surrogate_windows_p8_CR1 = surrogate_windows_p8_CR1.tolist()
viewing_windows_p8_CR1 = [
    (1714783700 - 1714775800, 1714787800 - 1714775800),
    (1714775800 - 1714775800, 1714787800 - 1714775800),
    (1714775900 - 1714775800, 1714783900 - 1714775800),
    (1714783900 - 1714775800, 1714783900 - 1714775800),
    (1714776000 - 1714775800, 1714784000 - 1714775800),
    (1714784100 - 1714775800, 1714776100 - 1714775800),
]

free_recall_windows_p8_FR2 = [
    [], # LA
    [6800,17000,25100,30100,46500], # attacks/bomb/bus/explosion
    [119800,472200], # white house/DC
    [174900,189800,200800,244200,301500,346800], # CIA/FBI
    [61500,59800,264700,289900,308400,454000], # hostage/exchange/sacrifice
    [292000], # handcuff/chair/tied
    [49800,55000,74300,82500,130200,261100,268400,290700,300300,323900,332400,337500,361800,374100,376600,412200,423700,421200,454700], # Jack Bauer
    [301100,329800], # Chloe
    [449500], # Bill
    [137300,143200,253000,266100,286800,350000,355000,373400,374000,411100], # Abu Fayed (main terrorist)
    [157500,164100,174900,143000,202700,205900,219000,228100,237700,388000,394800], # Ahmed Amar (kid)
    [59200,348600] # President
]
data = pd.read_csv('data/annotations_simulated/p8_FR2.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p8_FR2 = np.floor(data[1]).astype(int)
surrogate_windows_p8_FR2 = surrogate_windows_p8_FR2.tolist()

free_recall_windows_p8_CR2 = [
    [345500], # LA
    [128500,191300,344900], # attacks/bomb/bus/explosion
    [347400], # white house/DC
    [7800,66200,81900,196100,259800], # CIA/FBI
    [6500,245000,337600], # hostage/exchange/sacrifice
    [245200,257000,265900,272200,276500,286900,302500], # handcuff/chair/tied
    [5700,192600,242100,247700,246400,265300,222000,270900,289700,297100,298500,309300,316100,343900], # Jack Bauer
    [6800,14300], # Chloe
    [259200], # Bill
    [176600,194800,193700,199600,215200,177000], # Abu Fayed (main terrorist)
    [69300,72900,82100,352900,367300], # Ahmed Amar (kid)
    [144800,147500,163400] # President
]
data = pd.read_csv('data/annotations_simulated/p8_CR2.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p8_CR2 = np.floor(data[1]).astype(int)
surrogate_windows_p8_CR2 = surrogate_windows_p8_CR2.tolist()
viewing_windows_p8_CR2 = [
    (1714818800 - 1714818800, 1714810800 - 1714818800),
    (1714818800 - 1714818800, 1714810800 - 1714818800),
    (1714810900 - 1714818800, 1714810900 - 1714818800),
    (1714815000 - 1714818800, 1714815000 - 1714818800),
    (1714819000 - 1714818800, 1714815000 - 1714818800),
    (1714811100 - 1714818800, 1714811100 - 1714818800),
]

free_recall_windows_p9_FR1 = [
    [], # LA    
    [54000,170500], # attacks/bomb/bus/explosion
    [], # white house/DC
    [], # CIA/FBI
    [49500,52600,163400], # hostage/exchange/sacrifice
    [76600,99400], # handcuff/chair/tied
    [28100,32000,38200,69200,84400,93800,107800,165700,184400,187600], # Jack Bauer
    [], # Chloe
    [], # Bill
    # [80600,181300,1812500], # Abu Fayed (main terrorist)
    [80600, 83100, 181300,1812500], # Abu Fayed (main terrorist)
    [], # Ahmed Amar (kid)
    [145300] # President
]
data = pd.read_csv('data/annotations_simulated/p9_FR1.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p9_FR1 = np.floor(data[1]).astype(int)
surrogate_windows_p9_FR1 = surrogate_windows_p9_FR1.tolist()

free_recall_windows_p9_CR1 = [
    [], # LA
    [107000], # attacks/bomb/bus/explosion   
    [], # white house/DC
    [14100,47000], # CIA/FBI
    [272002,318600], # hostage/exchange/sacrifice
    [231300,236600], # handcuff/chair/tied
    [113300,162000,206400,239600,241300,249700,281200], # Jack Bauer
    [2200], # Chloe
    [220500,230500], # Bill
    [153700,178700,1810700,192100,290100,303800,311100], # Abu Fayed (main terrorist)
    [40400,57200,66900], # Ahmed Amar (kid)
    [104800,126300] # President
]
data = pd.read_csv('data/annotations_simulated/p9_CR1.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p9_CR1 = np.floor(data[1]).astype(int)
surrogate_windows_p9_CR1 = surrogate_windows_p9_CR1.tolist()
viewing_windows_p9_CR1 = [
    (1699300500 - 1699300600, 1699300600 - 1699300600),
    (1699300600 - 1699300600, 1699300600 - 1699300600),
    (1699300700 - 1699300600, 1699292700 - 1699300600),
    (1699292700 - 1699300600, 1699292700 - 1699300600),
    (1699300800 - 1699300600, 1699292800 - 1699300600),
    (1699300800 - 1699300600, 1699300800 - 1699300600),
]

free_recall_windows_p9_FR2 = [
    [], # LA
    [47200,106400,181900], # attacks/bomb/bus/explosion
    [], # white house/DC
    [], # CIA/FBI
    [45300,119200,179200], # hostage/exchange/sacrifice
    [], # handcuff/chair/tied
    [24900,48200,55200,64300,80200,116800,140600,151800,165800,201300,199600,208200], # Jack Bauer
    [], # Chloe
    [], # Bill
    [91100,101600,115600,148300,156800,169700,183700], # Abu Fayed (main terrorist)
    [], # Ahmed Amar (kid)
    [131300] # President
]
data = pd.read_csv('data/annotations_simulated/p9_FR2.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p9_FR2 = np.floor(data[1]).astype(int)
surrogate_windows_p9_FR2 = surrogate_windows_p9_FR2.tolist()

free_recall_windows_p9_CR2 = [
    [], # LA
    [212600,287300], # attacks/bomb/bus/explosion
    [], # white house/DC
    [], # CIA/FBI
    [16700,258300], # hostage/exchange/sacrifice
    [241600,720098], # handcuff/chair/tied
    [14600,192100,249000,268600,306300], # Jack Bauer
    [0,15500], # Chloe
    [], # Bill
    [20200,183800,192800,272300,279200,319100], # Abu Fayed (main terrorist)
    [], # Ahmed Amar (kid)
    [121700] # President
]
data = pd.read_csv('data/annotations_simulated/p9_CR2.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p9_CR2 = np.floor(data[1]).astype(int)
surrogate_windows_p9_CR2 = surrogate_windows_p9_CR2.tolist()    
viewing_windows_p9_CR2 = [
    (1699354100 - 1699354100, 1699346100 - 1699354100),
    (1699346100 - 1699354100, 1699354100 - 1699354100),
    (1699354200 - 1699354100, 1699354200 - 1699354100),
    (1699346200 - 1699354100, 1699354300 - 1699354100),
    (1699346300 - 1699354100, 1699346300 - 1699354100),
    (1699346400 - 1699354100, 1699346400 - 1699354100),
]

# 425000
offset_p10 = ((55*60 + 45) - (48*60 + 36)) * 5000
free_recall_windows_p10_FR1a = [
    [128600,421700], # LA
    [9400,20000,46600,321000], # attacks/bomb/bus/explosion
    [], # white house/DC
    [203200,212400], # CIA/FBI
    [84200,111300], # hostage/exchange/sacrifice
    [113500,378800], # handcuff/chair/tied
    [66200,81000,107400,146100,154200,176000,190700,213500,227800,227800,
        244400,349000,365600,366800,382000,402100], # Jack Bauer
    [], # Chloe
    [], # Bill
    [134700,167500,174300,173400,186700,237500,251100,322100,398200,416900], # Abu Fayed (main terrorist)
    [282400,300300,308500,324400], # Ahmed Amar (kid)
    [] # President
]
data = pd.read_csv('data/annotations_simulated/p10_FR1a.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p10_FR1a = np.floor(data[1]).astype(int)
surrogate_windows_p10_FR1a = surrogate_windows_p10_FR1a.tolist()

# patient did a second free recall before sleep!
free_recall_windows_p10_FR1b = [
    [202100,326500], # LA
    [900,16900,40000,88200,110000], # attacks/bomb/bus/explosion
    [], # white house/DC
    [56600,244500,313000,362000], # CIA/FBI
    [104500,152900], # hostage/exchange/sacrifice
    [213300], # handcuff/chair/tied
    [77600,912100,110100,115500,127600,143200,164000,172700,179700,179800,201400,205600,219600,229200,256000,268600,272500,295400], # Jack Bauer
    [], # Chloe
    [], # Bill
    [49700,86100,214800,300000,302600,468000,484600], # Abu Fayed (main terrorist)
    [346200,351200,372300,381900,395000,413200,430300,430100,442700,464400,472000,469400], # Ahmed Amar (kid)
    [] # President
]
data = pd.read_csv('data/annotations_simulated/p10_FR1b.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p10_FR1b = np.floor(data[1]).astype(int)
surrogate_windows_p10_FR1b = surrogate_windows_p10_FR1b.tolist()

free_recall_windows_p10_FR1 = [fr1a + [fr1b_item + offset_p10 for fr1b_item in fr1b]  
                                for fr1a, fr1b in zip(free_recall_windows_p10_FR1a, free_recall_windows_p10_FR1b)]
surrogate_windows_p10_FR1 = surrogate_windows_p10_FR1a + [fr1b_item + offset_p10 for fr1b_item in surrogate_windows_p10_FR1b]

free_recall_windows_p10_CR1 = [
    [287700], # LA
    [68400], # attacks/bomb/bus/explosion
    [332400,466500], # white house/DC
    [6300,325800,455400,462100], # CIA/FBI
    [132200,257000], # hostage/exchange/sacrifice
    [235900], # handcuff/chair/tied
    [4900,123400,1211500,132900,159700,212100,243000,251300,252000,263700,476500], # Jack Bauer
    [0,461700,475800,482500], # Chloe
    [], # Bill
    [179700,190900,214300], # Abu Fayed (main terrorist)
    [75600], # Ahmed Amar (kid)
    [116400,139500,158600,156700] # President
]
data = pd.read_csv('data/annotations_simulated/p10_CR1.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p10_CR1 = np.floor(data[1]).astype(int)
surrogate_windows_p10_CR1 = surrogate_windows_p10_CR1.tolist()
viewing_windows_p10_CR1 = [
    (1702326200 - 1702326200, 1702326200 - 1702326200),
    (1702318200 - 1702326200, 1702326200 - 1702326200),
    (1702326300 - 1702326200, 1702318300 - 1702326200),
    (1702326300 - 1702326200, 1702318400 - 1702326200),
    (1702318400 - 1702326200, 1702326400 - 1702326200),
    (1702318500 - 1702326200, 1702326500 - 1702326200),
]

free_recall_windows_p10_FR2 = [
    [24600,236300,444900], # LA
    [8900,24200,31800,107000,450], # attacks/bomb/bus/explosion
    [], # white house/DC
    [276100,309400,313900,475700,504300], # CIA/FBI
    [56700,85600,99600,120700,172700], # hostage/exchange/sacrifice
    [243600,415800], # handcuff/chair/tied
    [67100,70000,75700,85100,92500,100100,120100,136700,181100,191100,189700,207200,214200,244300,253300,269400,278700,309300,331400,350600,357200,359100,370100,387600,394600,409600,424700,613300], # Jack Bauer
    [255100], # Chloe
    [], # Bill
    [49300,159500,296600,336700,355600,361100,359100,606800,616800], # Abu Fayed (main terrorist)
    [478400,508900,517900,523100,529400,553000,564500,579200,582800,598600,610700,626200], # Ahmed Amar (kid)
    [110000,190000,261500] # President
]
data = pd.read_csv('data/annotations_simulated/p10_FR2.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p10_FR2 = np.floor(data[1]).astype(int)
surrogate_windows_p10_FR2 = surrogate_windows_p10_FR2.tolist()

free_recall_windows_p10_CR2 = [
    [314600], # LA
    [160100,193400,301100], # attacks/bomb/bus/explosion
    [326500], # white house/DC
    [77700,321500], # CIA/FBI
    [], # hostage/exchange/sacrifice
    [], # handcuff/chair/tied
    [6900,14200,121300,127100,140800,153400,160700,183400,207400,218000,244900,247800,254600,271800], # Jack Bauer
    [0], # Chloe
    [], # Bill
    [185400,203300,209600,207000], # Abu Fayed (main terrorist)
    [74100,82400], # Ahmed Amar (kid)
    [130600,129200,137100,142900,149200] # President
]
data = pd.read_csv('data/annotations_simulated/p10_CR2.ann', sep='^([^\s]*)\s', engine='python', header=None)
data[1] = pd.to_numeric(data[1], errors='coerce')
data.dropna(subset=[1], inplace=True)
surrogate_windows_p10_CR2 = np.floor(data[1]).astype(int)
surrogate_windows_p10_CR2 = surrogate_windows_p10_CR2.tolist()
viewing_windows_p10_CR2 = [
    (1702391100 - 1702399100, 1702391200 - 1702399100),
    (1702399200 - 1702399100, 1702391200 - 1702399100),
    (1702391300 - 1702399100, 1702399300 - 1702399100),
    (1702399300 - 1702399100, 1702399300 - 1702399100),
    (1702391400 - 1702399100, 1702399400 - 1702399100),
    (1702391400 - 1702399100, 1702391400 - 1702399100),
]


def find_target_activation_indices(time, concept_vocalz_msec, win_range_bins, end_inclusive=True):
    concept_vocalz_bin = []
    target_activations_indices = []
    if len(concept_vocalz_msec) > 0:  # if person said the concept at all
        for concept_vocalization in concept_vocalz_msec:  # get the indices for each mention
            concept_vocalization_bin = np.abs(time - concept_vocalization/5000).argmin()
            win_edge_1 = concept_vocalization_bin + win_range_bins[0]
            win_edge_2 = concept_vocalization_bin + win_range_bins[1]
            if end_inclusive:
                win_edge_2 += 1
            if win_edge_1 >= 0 and win_edge_2 < len(time):  # only take vocalizations where full window can be obtained
                target_activations_indices.append(np.arange(win_edge_1, win_edge_2, dtype=int))
                concept_vocalz_bin.append(concept_vocalization_bin)
    else:
        concept_vocalz_bin = []
        target_activations_indices = []
    return concept_vocalz_bin, target_activations_indices

