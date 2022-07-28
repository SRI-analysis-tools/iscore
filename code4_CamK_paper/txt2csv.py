import numpy as np
import pandas as pd
import os
#dirtxt=r'C:\Users\US Retail\Documents\camkpaper\processed_data\camK_dosing'
#dirtxt=r'C:\Users\US Retail\Documents\camkpaper\Peng_VG2\txtfiles'
#dirtxt=r'C:\Users\US Retail\Documents\camkpaper\processed_data\mats_gad_cre_camK_dta\dosing'
dirtxt=r'C:\Users\US Retail\Documents\camkpaper\processed_data\mats_gad_cre_camK_dta\dosing\txt_cre'
txtfiles = os.listdir(dirtxt)
txtfiles=[f for f in txtfiles if f.endswith('.txt')]
#convert all txt files to csv
for fntxt in txtfiles:
    varn = fntxt[4:-4]
    fn=os.path.join(dirtxt,fntxt)
    cols=['subjectID',varn,'ZT','Cond']
    df=pd.DataFrame(columns=cols)
    with open(fn) as f:
            lines = f.readlines()
    posc=0
    #cnds=['ALM-CNO','ALM-SAL','VEH-CNO','VEH-SAL']
    cnds=['SAL','CNO']
    firstZT=3
    for i,l in enumerate(lines):
        df1l = pd.DataFrame(columns=cols)
        numb=[float(e) for e in l.split()]
        df1l[cols[1]] = numb
        df1l[cols[2]] = np.arange(firstZT,firstZT+len(numb))
        df1l[cols[0]] = 'mse'+str(1+i//len(cnds))
        df1l[cols[-1]]=cnds[posc]
        posc+=1
        if posc>len(cnds)-1:
            posc=0
        df=df.append(df1l,ignore_index=True)
    df.to_excel(os.path.join(dirtxt,fntxt[:-3]+'xls'),index=False)
