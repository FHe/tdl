{'application':{'type':'Application',
          'name':'Template',
    'backgrounds': [
    {'type':'Background',
          'name':'wxXRF_bgTemplate',
          'title':u'Spec Data',
          'size':(584, 786),
          'statusBar':1,
          'style':['resizeable'],

        'menubar': {'type':'MenuBar',
         'menus': [
             {'type':'Menu',
             'name':'menuFile',
             'label':'&File',
             'items': [
                  {'type':'MenuItem',
                   'name':'menuFileReadXrf',
                   'label':u'ReadXrfFile',
                  },
                  {'type':'MenuItem',
                   'name':'menuFileSaveScript',
                   'label':'SaveScript',
                  },
                  {'type':'MenuItem',
                   'name':'menuFileLoadScript',
                   'label':'LoadScript',
                  },
                  {'type':'MenuItem',
                   'name':'menuFileExit',
                   'label':'E&xit',
                  },
              ]
             },
         ]
     },
         'components': [

{'type':'StaticText', 
    'name':'addBadPixelFileRois', 
    'position':(49, 467), 
    'text':'Add bad pixel file, rois', 
    },

{'type':'StaticText', 
    'name':'colormap', 
    'position':(455, 614), 
    'text':'colormap', 
    },

{'type':'Choice', 
    'name':'ColorMap', 
    'position':(433, 629), 
    'size':(105, -1), 
    'items':[], 
    },

{'type':'StaticBox', 
    'name':'StaticBox5', 
    'position':(6, 160), 
    'size':(558, 56), 
    'label':'Scan Variable', 
    },

{'type':'Button', 
    'name':'ImgPathSel', 
    'position':(509, 425), 
    'size':(48, 22), 
    'label':'Select', 
    },

{'type':'TextField', 
    'name':'ImagePath', 
    'position':(93, 424), 
    'size':(400, -1), 
    },

{'type':'StaticText', 
    'name':'text', 
    'position':(19, 428), 
    'text':'Image Path', 
    'toolTip':'Provide Tdl variable hodling values', 
    },

{'type':'Button', 
    'name':'Update', 
    'position':(16, 7), 
    'size':(56, 22), 
    'label':'Update', 
    },

{'type':'StaticText', 
    'name':'Lbl1', 
    'position':(86, 11), 
    'text':'ReaderGroup', 
    },

{'type':'ComboBox', 
    'name':'Grp', 
    'position':(168, 7), 
    'size':(141, -1), 
    'items':[], 
    },

{'type':'StaticText', 
    'name':'Lbl2', 
    'position':(315, 11), 
    'text':'ReaderName', 
    },

{'type':'ComboBox', 
    'name':'Reader', 
    'position':(398, 7), 
    'size':(155, -1), 
    'items':[u'reader'], 
    'stringSelection':'reader', 
    'text':'reader', 
    },

{'type':'StaticBox', 
    'name':'StaticBox21', 
    'position':(5, 35), 
    'size':(557, 120), 
    'label':'Spec File', 
    },

{'type':'StaticText', 
    'name':'StaticText1', 
    'position':(20, 63), 
    'text':'Spec Path', 
    },

{'type':'TextField', 
    'name':'SpecPath', 
    'position':(91, 60), 
    'size':(400, -1), 
    },

{'type':'Button', 
    'name':'SpecPathSel', 
    'position':(499, 60), 
    'size':(48, 22), 
    'label':'Select', 
    },

{'type':'StaticText', 
    'name':'Lbl111', 
    'position':(19, 95), 
    'text':'Spec File(s)', 
    },

{'type':'Choice', 
    'name':'SpecFile', 
    'position':(91, 91), 
    'size':(273, -1), 
    'items':[], 
    },

{'type':'Button', 
    'name':'SpecFileSel', 
    'position':(374, 91), 
    'size':(48, 22), 
    'label':'Select', 
    },

{'type':'StaticText', 
    'name':'Lbl11', 
    'position':(20, 126), 
    'text':'Scan Num', 
    },

{'type':'ComboBox', 
    'name':'ScanNum', 
    'position':(91, 122), 
    'size':(78, -1), 
    'items':[], 
    },

{'type':'CheckBox', 
    'name':'ReadMed', 
    'position':(182, 124), 
    'size':(78, 20), 
    'label':'Read Med', 
    },

{'type':'CheckBox', 
    'name':'ReadXrf', 
    'position':(276, 126), 
    'label':'Read XRF', 
    },

{'type':'CheckBox', 
    'name':'ReadImg', 
    'position':(369, 126), 
    'size':(89, -1), 
    'label':'Read Image', 
    },

{'type':'Button', 
    'name':'Read', 
    'position':(469, 121), 
    'size':(46, 25), 
    'label':'Read', 
    },

{'type':'StaticText', 
    'name':'Lbl3', 
    'position':(24, 182), 
    'text':'Variable Name', 
    },

{'type':'ComboBox', 
    'name':'ScanVar', 
    'position':(124, 178), 
    'size':(186, -1), 
    'items':[], 
    },

{'type':'CheckBox', 
    'name':'AutoUpdateCheck', 
    'position':(319, 182), 
    'size':(72, 15), 
    'checked':True, 
    'label':'Auto Plot', 
    },

{'type':'CheckBox', 
    'name':'AutoCalcVarName', 
    'position':(406, 182), 
    'size':(74, -1), 
    'label':'Auto Calc', 
    },

{'type':'CheckBox', 
    'name':'LongName', 
    'position':(494, 182), 
    'label':'Long', 
    },

{'type':'StaticBox', 
    'name':'StaticBox2', 
    'position':(7, 223), 
    'size':(559, 175), 
    'label':'MED Params', 
    },

{'type':'StaticText', 
    'name':'medxrfPath', 
    'position':(12, 249), 
    'text':'Med/XRF Path', 
    'toolTip':'Provide Tdl variable hodling values', 
    },

{'type':'TextField', 
    'name':'MedPath', 
    'position':(97, 245), 
    'size':(400, -1), 
    },

{'type':'Button', 
    'name':'MedPathSel', 
    'position':(508, 246), 
    'size':(48, 22), 
    'label':'Select', 
    },

{'type':'StaticText', 
    'name':'Lbl1001', 
    'position':(18, 297), 
    'text':'XRF Lines =', 
    'toolTip':'Provide Tdl variable hodling values', 
    },

{'type':'ComboBox', 
    'name':'XrfLines', 
    'position':(99, 291), 
    'size':(153, -1), 
    'items':[], 
    },

{'type':'CheckBox', 
    'name':'Align', 
    'position':(263, 294), 
    'checked':True, 
    'label':'Align', 
    },

{'type':'CheckBox', 
    'name':'Total', 
    'position':(263, 359), 
    'checked':True, 
    'label':'Total', 
    },

{'type':'StaticText', 
    'name':'McaListLbl', 
    'position':(20, 327), 
    'text':'Bad Mcas =', 
    'toolTip':'Provide Tdl variable hodling values', 
    },

{'type':'ComboBox', 
    'name':'BadMcas', 
    'position':(99, 325), 
    'size':(153, -1), 
    'items':[], 
    },

{'type':'CheckBox', 
    'name':'CorrectData', 
    'position':(263, 327), 
    'checked':True, 
    'label':'Correct', 
    },

{'type':'StaticText', 
    'name':'EminLbl', 
    'position':(11, 362), 
    'text':'Emin (keV) =', 
    },

{'type':'TextField', 
    'name':'Emin', 
    'position':(99, 358), 
    'size':(46, -1), 
    'text':'-1', 
    },

{'type':'StaticText', 
    'name':'TauListLbl', 
    'position':(351, 302), 
    'text':'Taus =', 
    'toolTip':'Provide Tdl variable holding Tau values', 
    },

{'type':'ComboBox', 
    'name':'McaTaus', 
    'position':(402, 298), 
    'size':(153, -1), 
    'items':[], 
    },

{'type':'StaticText', 
    'name':'EmaxLbl', 
    'position':(149, 362), 
    'text':'Emax =', 
    },

{'type':'TextField', 
    'name':'Emax', 
    'position':(199, 358), 
    'size':(50, -1), 
    'text':'-1', 
    },

{'type':'StaticBox', 
    'name':'StaticBox1', 
    'position':(333, 274), 
    'size':(233, 124), 
    'label':'Dead Time', 
    },

{'type':'Button', 
    'name':'FitDeadtime', 
    'position':(338, 342), 
    'size':(33, 39), 
    'label':'Fit', 
    },

{'type':'StaticText', 
    'name':'StaticText32', 
    'position':(398, 338), 
    'size':(20, 15), 
    'text':'X=', 
    },

{'type':'Choice', 
    'name':'DTX', 
    'position':(419, 334), 
    'size':(136, -1), 
    'items':[], 
    },

{'type':'StaticText', 
    'name':'StaticText321', 
    'position':(378, 368), 
    'text':'Norm=', 
    },

{'type':'Choice', 
    'name':'DTNorm', 
    'position':(419, 365), 
    'size':(136, -1), 
    'items':[], 
    },

{'type':'StaticBox', 
    'name':'StaticBox4', 
    'position':(9, 396), 
    'size':(554, 108), 
    'label':'Image Params', 
    },

{'type':'StaticBox', 
    'name':'StaticBox3', 
    'position':(8, 518), 
    'size':(309, 157), 
    'label':'Plot Scalers', 
    },

{'type':'Button', 
    'name':'PlotScaler', 
    'position':(24, 543), 
    'size':(41, 31), 
    'label':'Plot', 
    },

{'type':'CheckBox', 
    'name':'ScalerPlot', 
    'position':(85, 549), 
    'checked':True, 
    'label':'Auto Plot Scalers', 
    },

{'type':'CheckBox', 
    'name':'DefaultScalerAxes', 
    'position':(214, 549), 
    'checked':True, 
    'label':'Default Axes', 
    },

{'type':'StaticBox', 
    'name':'plotMedimage', 
    'position':(325, 519), 
    'size':(239, 155), 
    'label':'Plot Med/Image', 
    },

{'type':'StaticText', 
    'name':'StaticText2', 
    'position':(343, 548), 
    'text':'Scan Point', 
    },

{'type':'ComboBox', 
    'name':'ScanPnt', 
    'position':(418, 545), 
    'size':(61, -1), 
    'items':[], 
    'text':'0', 
    },

{'type':'StaticText', 
    'name':'StaticText3', 
    'position':(47, 586), 
    'text':'X=', 
    },

{'type':'Choice', 
    'name':'XPlot', 
    'position':(70, 582), 
    'size':(183, -1), 
    'items':[], 
    },

{'type':'CheckBox', 
    'name':'XlogCheck', 
    'position':(265, 586), 
    'size':(45, -1), 
    'label':'Xlog', 
    },

{'type':'StaticText', 
    'name':'StaticText31', 
    'position':(48, 615), 
    'text':'Y=', 
    },

{'type':'Choice', 
    'name':'YPlot', 
    'position':(70, 611), 
    'size':(183, -1), 
    'items':[], 
    },

{'type':'CheckBox', 
    'name':'YlogCheck', 
    'position':(265, 615), 
    'label':'Ylog', 
    },

{'type':'Button', 
    'name':'PlotMed', 
    'position':(340, 586), 
    'size':(76, -1), 
    'label':'Plot Med', 
    },

{'type':'CheckBox', 
    'name':'MedYlog', 
    'position':(439, 591), 
    'label':'Ylog', 
    },

{'type':'CheckBox', 
    'name':'MedHold', 
    'position':(494, 592), 
    'size':(43, -1), 
    'label':'Hold', 
    },

{'type':'StaticText', 
    'name':'StaticText311', 
    'position':(24, 644), 
    'text':'Norm=', 
    },

{'type':'Choice', 
    'name':'NormPlot', 
    'position':(70, 640), 
    'size':(183, -1), 
    'items':[], 
    },

{'type':'CheckBox', 
    'name':'HoldCheck', 
    'position':(265, 644), 
    'label':'Hold', 
    },

{'type':'Button', 
    'name':'PlotImg', 
    'position':(340, 628), 
    'size':(76, -1), 
    'label':'Plot Image', 
    },

{'type':'StaticLine', 
    'name':'StaticLine1', 
    'position':(10, 683), 
    'size':(559, 6), 
    'layout':'horizontal', 
    },

] # end components
} # end background
] # end backgrounds
} }
