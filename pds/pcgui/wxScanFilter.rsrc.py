{'application':{'type':'Application',
          'name':'Template',
    'backgrounds': [
    {'type':'Background',
          'name':'bgTemplate',
          'title':u'Scan Filters',
          'size':(832, 556),
          'style':['resizeable'],

         'components': [

{'type':'Button', 
    'name':'AppendFiltered', 
    'position':(700, 488), 
    'size':(105, -1), 
    'label':u'Append Scans', 
    },

{'type':'Button', 
    'name':'SaveFiltered', 
    'position':(580, 488), 
    'size':(105, -1), 
    'label':u'New Integrator', 
    },

{'type':'StaticText', 
    'name':'FileName', 
    'position':(570, 16), 
    'text':u'File name:', 
    },

{'type':'Button', 
    'name':'FilterLs', 
    'position':(127, 8), 
    'size':(121, 32), 
    'label':u'Filter L Range', 
    },

{'type':'Button', 
    'name':'NumberFilter', 
    'position':(8, 8), 
    'size':(42, 32), 
    'label':u'Filter\n#s', 
    },

{'type':'Button', 
    'name':'ResetButton', 
    'position':(393, 491), 
    'label':u'Reset Filters and Reread Data', 
    },

{'type':'Button', 
    'name':'TypeFilter', 
    'position':(247, 8), 
    'size':(67, 32), 
    'label':u'Filter Types', 
    },

{'type':'TextArea', 
    'name':'MoreInfoHere', 
    'position':(579, 74), 
    'size':(222, 327), 
    'editable':False, 
    'text':u'More Info', 
    },

{'type':'StaticLine', 
    'name':'StaticLine6', 
    'position':(570, 478), 
    'size':(231, -1), 
    'layout':'horizontal', 
    },

{'type':'Button', 
    'name':'DateFilter', 
    'position':(369, 8), 
    'size':(176, 32), 
    'label':u'Filter Date Range', 
    },

{'type':'StaticLine', 
    'name':'StaticLine5', 
    'position':(565, 490), 
    'size':(2, 24), 
    'layout':'horizontal', 
    },

{'type':'Button', 
    'name':'Update', 
    'position':(8, 491), 
    'size':(105, -1), 
    'label':u'Keep Selected', 
    'toolTip':u'Discard all scans not currently\nselected in the above window.', 
    },

{'type':'StaticLine', 
    'name':'StaticLine4', 
    'position':(570, 38), 
    'size':(231, -1), 
    'layout':'horizontal', 
    },

{'type':'StaticText', 
    'name':'MoreInfo', 
    'position':(574, 52), 
    'text':u'More Info:', 
    },

{'type':'Button', 
    'name':'HKFilter', 
    'position':(49, 8), 
    'size':(79, 32), 
    'label':u'Filter HKs', 
    },

{'type':'MultiColumnList', 
    'name':'unfilteredList', 
    'position':(8, 44), 
    'size':(556, 441), 
    'backgroundColor':(255, 255, 255, 255), 
    'columnHeadings':['#', 'H Val', 'K Val', 'L Start', 'L Stop', 'Scan Type', 'Aborted', 'Date'], 
    'font':{'faceName': u'Segoe UI', 'family': 'sansSerif', 'size': 9}, 
    'items':[], 
    'maxColumns':8, 
    'rules':True, 
    },

] # end components
} # end background
] # end backgrounds
} }
