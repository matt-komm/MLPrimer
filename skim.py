import uproot
import pandas as pd
import numpy as np
import awkward as ak
import sklearn.decomposition

import matplotlib.pyplot as plt


processedData = {
    'jet_img': [],
    'jet_pt': [],
    'jet_eta': [],
    'jet_nparticles': [],
    'jet_sdmass': [],
    'jet_tau1': [],
    'jet_tau2': [],
    'jet_tau3': [],
    'jet_tau4': [],
    'jet_charge': [],
    
    'label_Zqq': [],
    'label_Wqq': [],
    'label_Tbqq': [],
    
    'true_pt': [],
}
binning = [np.linspace(-0.8,0.8,21),np.linspace(-0.8,0.8,21)]

for fileName in ['TTBar_120.root','ZToQQ_120.root','WToQQ_120.root']:
    f = uproot.open(fileName)
    #print (f['tree'].keys())

    


    n = 0

    for data in f['tree'].iterate(
        [
            'part_deta',
            'part_dphi',
            'part_energy',
            'part_charge',
            
            'part_d0val',
            'part_d0err',
            
            'part_px',
            'part_py',
            
            'jet_pt',
            'jet_eta',
            'jet_nparticles',
            'jet_sdmass',
            'jet_tau1',
            'jet_tau2',
            'jet_tau3',
            'jet_tau4',
            
            'label_Zqq',
            'label_Wqq',
            'label_Tbqq',
            
            'aux_genpart_pt',
        ],
            step_size=100,library='np'
    ):
        n+=len(data['jet_pt'])
        if n>20000:
            break
        
        print (fileName,n,'/',f['tree'].num_entries,100.0*n/f['tree'].num_entries)
        for iev in range(len(data['jet_pt'])):
        
            deta = data['part_deta'][iev]
            dphi = data['part_dphi'][iev]
            pt = np.sqrt(data['part_px'][iev]**2+data['part_py'][iev]**2)
            charge = data['part_px'][iev]
            dxySig = np.log(1+np.abs(data['part_d0val'][iev])/(1e-4+np.abs(data['part_d0err'][iev])))
            
            beta = 0.6
            jet_charge = np.sum((pt**beta)*charge)/np.sum(pt**beta)
            processedData['jet_charge'].append(jet_charge)
            
            coords = np.stack([deta,dphi],axis=1)
            #print (coords)
            
            pca = sklearn.decomposition.PCA(n_components=2)
            pca.fit(coords)
            
            angle = np.arctan2(pca.components_[0,1],pca.components_[0,0])
            
            rotMatrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)],
            ])
            
            rotCoords = np.dot(coords,rotMatrix)
            rotCoords[:,1] *= np.dot(pca.components_,rotMatrix)[1,1]

            #pca2 = sklearn.decomposition.PCA(n_components=2)
            #pca2.fit(rotCoords)
            #print(pca2.components_)
            #print (rotCoords)
            
            histPt, _, _ = np.histogram2d(rotCoords[:,0],rotCoords[:,1],bins=binning,weights=pt)
            histCharge, _, _ = np.histogram2d(rotCoords[:,0],rotCoords[:,1],bins=binning,weights=np.abs(charge))
            histDxy, _, _ = np.histogram2d(rotCoords[:,0],rotCoords[:,1],bins=binning,weights=dxySig)
            
            hist = np.stack([histPt,histCharge,histDxy],axis=2)
            
            processedData['jet_img'].append(hist)
            
            
            for feature in [
                'jet_pt',
                'jet_eta',
                'jet_nparticles',
                'jet_sdmass',
                'jet_tau1',
                'jet_tau2',
                'jet_tau3',
                'jet_tau4',
                
                'label_Zqq',
                'label_Wqq',
                'label_Tbqq',
            ]:
                processedData[feature].append(data[feature][iev])
            processedData['true_pt'].append(data['aux_genpart_pt'][iev])
           

for k in processedData.keys():
    processedData[k] = np.stack(processedData[k],axis=0).astype(np.float32)
    print (k,processedData[k].shape)
    
rndIdx = np.arange(0,processedData['jet_pt'].shape[0])
np.random.shuffle(rndIdx)

for k in processedData.keys():
    processedData[k] = processedData[k][rndIdx]

np.savez_compressed("data.npz",**processedData)
    

