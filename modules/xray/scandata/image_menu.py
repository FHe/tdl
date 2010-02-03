#######################################################################
"""
Tom Trainor (fftpt@uaf.edu)

Menu function to handle interactive
image processing

Modifications:
--------------


"""
#######################################################################

import types
import copy
import numpy as num
from matplotlib import pyplot

from   pds.shellutil import Menu, show_more 
from   pds.shellutil import get_tf, get_yn, get_int, get_flt
from   plotter import cursor
import image_data
import data

########################################################################
IMG_HEADER = """
Number of images  = %s
Current image     = %s
Current image roi = %s
Current image rotation angle = %s
"""

IMG_LABELS = ['display','imax','rotangle','setroi','plotsums',
              'selectroi','bgr','copyall','integrate','intall',
              'point','next','previous','flag','help','quit']
IMG_DESCR = ["Display image",
             "Set max image intensity value",
             "Set image rotation angle (deg ccw)",
             "Set roi from image zoom (Figure 1)",
             "Plot row/column sums (Figure 2)",
             "Select roi from sum plots (Figure 2)",
             "Set background parameters",
             "Apply current roi and background params to all images",
             "Integrate current image",
             "Integrate all images",
             "Select scan point",
             "Select next point ",
             "Select previous point", 
             "Flag as bad point",
             "Show options",
             "Quit / All Done"]

########################################################################
def image_menu(imdata):
    """
    Interactively inspect/integrate images in ScanData or a
    ImageScan object
    """
    if isinstance(imdata,data.ScanData):
        imdata = imdata.image
    elif isinstance(imdata,image_data.ImageScan):
        pass
    else:
        print "Invalid image data"
    prompt   = 'Select option >'
    npts     = len(imdata.image)
    scan_pnt = 0
    roi      = []
    ret      = ''
    im_max   = None

    # local plot fun
    def _implot(imdata,scan_pnt):
        rotangle = imdata.rotangle[scan_pnt]
        im_max = imdata.im_max[scan_pnt]
        figtitle = "Scan Point = %i" % (scan_pnt)
        image_data.image_plot(imdata.image[scan_pnt],fig=1,verbose=True,
                              figtitle=figtitle,im_max=im_max,rotangle=rotangle)

    # check init and plot first
    if imdata._is_init() == False:
        imdata._init_image()
    _implot(imdata,scan_pnt)
    
    # make menu
    m = Menu(labels=IMG_LABELS,descr=IMG_DESCR,sort=False,matchidx=True)
    
    # loop
    while ret != 'quit':
        roi      = imdata.rois[scan_pnt]
        rotangle = imdata.rotangle[scan_pnt]
        header   = IMG_HEADER % (str(npts),str(scan_pnt),str(roi),str(rotangle))
        m.header = header
        ret      = m.prompt(prompt)

        if ret == 'display':
            _implot(imdata,scan_pnt)
        elif ret == 'imax':
            print 'Image max intensity = ', imdata.image[scan_pnt].max()
            im_max = get_int(prompt='Enter maximum intensity value for image plot',
                             default=imdata.im_max[scan_pnt],min=-1)
            imdata.im_max[scan_pnt] = im_max
            _implot(imdata,scan_pnt)
        elif ret == "rotangle":
            rotangle = get_flt(prompt='Enter rotation angle in degrees ccw',
                               default=imdata.rotangle[scan_pnt],
                               min=-360.,max=360.)
            imdata.rotangle[scan_pnt] = rotangle
            _implot(imdata,scan_pnt)
        elif ret == 'setroi':
            pyplot.figure(1)
            (x1,x2,y1,y2) = pyplot.axis()
            roi  = [int(x1),int(y1),int(x2),int(y2)]
            imdata.rois[scan_pnt] = roi
        elif ret == 'plotsums':
            roi      = imdata.rois[scan_pnt]
            rotangle = imdata.rotangle[scan_pnt]
            image    = image_data.clip_image(imdata.image[scan_pnt],roi,
                                             rotangle=rotangle)
            bgr_par  = imdata.bgrpar[scan_pnt]
            image_data.sum_plot(image,fig=2,**bgr_par)
        elif ret == 'selectroi':
            image = imdata.image[scan_pnt]
            bgr_par = imdata.bgrpar[scan_pnt]
            image_data.sum_plot(image,fig=2,**bgr_par)
            c = cursor(fig=2)
            (c1,y) = c.get_click(msg="Select left col sum")
            (c2,y) = c.get_click(msg="Select right col sum")
            (r1,y) = c.get_click(msg="Select left row sum")
            (r2,y) = c.get_click(msg="Select right row sum")
            roi = [int(c1),int(r1),int(c2),int(r2)]
            imdata.rois[scan_pnt] = roi
        elif ret == 'bgr':
            bgr_par = imdata.bgrpar[scan_pnt]
            bgr_par = bgr_menu(bgr_par)
            imdata.bgrpar[scan_pnt] = bgr_par
        elif ret == 'copyall':
            roi      = imdata.rois[scan_pnt]
            rotangle = imdata.rotangle[scan_pnt]
            bgr_par  = imdata.bgrpar[scan_pnt]
            #
            imdata.rois = []
            imdata.rotangle = []
            imdata.bgrpar = []
            #
            for j in range(npts):
                imdata.rois.append(copy.copy(roi))
                imdata.rotangle.append(rotangle)
                imdata.bgrpar.append(copy.copy(bgr_par))
        elif ret == 'integrate':
            imdata.integrate(idx=[scan_pnt],
                                   plot=True,fig=3)
        elif ret == 'intall':
            yn = get_tf("Plot all images",default=False)
            imdata.integrate(plot=yn)
            #
            pyplot.figure(5, figsize = [5,4])
            pyplot.clf()
            #
            x = num.arange(len(imdata.image))
            pyplot.plot(x,imdata.peaks['I'],'b',label='image sum')
            pyplot.errorbar(x,imdata.peaks['I'],imdata.peaks['Ierr'],fmt='bo')
            #
            pyplot.plot(x,imdata.peaks['I_c'],'r',label='col sum')
            pyplot.errorbar(x,imdata.peaks['I_c'],imdata.peaks['Ierr_c'],fmt='ro')
            #
            pyplot.plot(x,imdata.peaks['I_r'],'g',label='row sum')
            pyplot.errorbar(x,imdata.peaks['I_r'],imdata.peaks['Ierr_r'],fmt='go')
            #
            pyplot.semilogy()
            pyplot.legend(loc = 9)
            pyplot.xlabel('Point')
            pyplot.ylabel('Integrated Intensity')
        elif ret == 'point':
            scan_pnt = get_int(prompt='Enter scan point',
                               default=scan_pnt,min=0,max = npts-1)
            _implot(imdata,scan_pnt)
        elif ret == 'next':
            if scan_pnt + 1 < npts: 
                scan_pnt = scan_pnt + 1
                _implot(imdata,scan_pnt)
        elif ret == 'previous':
            if scan_pnt - 1 > -1: 
                scan_pnt = scan_pnt - 1
                _implot(imdata,scan_pnt)
        elif ret == 'flag':
            if int(scan_pnt) in imdata.bad_points:
                imdata.bad_points.remove(scan_pnt)
                print "Data point removed from bad list"
            else:
                imdata.bad_points.append(int(scan_pnt))
                print "Data point added to bad list"
        else:
            pass

########################################################################
BGR_INFO = """
################################################
* bgrflag is flag for how to do backgrounds:
   = 0 determine row and column backgrounds after summation
   = 1 determine 2D background using fits to the 'c'olumn direction 
   = 2 determine 2D background using fits to the 'r'ow direction 
   = 3 determine 2D background using average of the
       'r'ow and 'c'olumn direction fits 

  ==> below params are for 'c'olumn and 'r'ow directions

* c/rnbgr = number of end points to use in linear background determination
  (see background.background)
      
* c/rwidth should correspond roughly to the actual peak widths
  The background function should fit features that are in
  general broader than these values
    Estimate cwidth using the width of the peak in the row sum.
    Estimate rwidth using the width of the peak in the col sum.
  Note that width = 0 corresponds to no polynomial background
  
* c/rpow is the power of the polynomial used in background determination
  (see background.background)
  
* c/rtangent is a flag to indicate if local slope of the data should be fitted 
  (see background.background)
################################################
"""

BGR_HEADER = """
* bgrflag=%i,
* cnbgr=%i, cwidth=%g,cpow=%g,ctan=%s
* rnbgr=%i, rwidth=%g,rpow=%g,rtan=%s
"""

BGR_LABELS = ['help','info','bgrflag',
              'cnbgr','cwidth','cpow','ctan',
              'rnbgr','rwidth','rpow','rtan',
              'quit']

BGR_DESCR = ["Show options","Get more info on parameter defintions",
             "Set background flag",
             "Set num bgr for linear background - column direction",
             "Set peak width for non-linear background - column direction",
             "Set polynomial power for non-linear background - column direction",
             "Set tangent flag (True or False) - column direction",
             "Set num bgr for linear background - row direction",
             "Set peak width for non-linear background - row direction",
             "Set polynomial power for non-linear background - row direction",
             "Set tangent flag (True or False) - row direction",
             "Quit / All done"]

IMG_BGR_PARAMS = {'bgrflag':0,
                  'cnbgr':5,'cwidth':0,'cpow':2.,'ctan':False,
                  'rnbgr':5,'rwidth':0,'rpow':2.,'rtan':False}

def bgr_menu(bgr_params=IMG_BGR_PARAMS):
    """
    Get bgr options
    """
    prompt   = 'Select option >'

    # make menu
    m   = Menu(labels=BGR_LABELS,descr=BGR_DESCR,sort=False,matchidx=True)
    ret = ''
    
    while ret != 'quit':
        header = BGR_HEADER % (bgr_params['bgrflag'],
                               bgr_params['cnbgr'],bgr_params['cwidth'],
                               bgr_params['cpow'],str(bgr_params['ctan']),
                               bgr_params['rnbgr'],bgr_params['rwidth'],
                               bgr_params['rpow'],str(bgr_params['rtan']))
        m.header = header
        ret      = m.prompt(prompt)
        #
        if ret == 'bgrflag':
            bgr_params['bgrflag'] = get_int(prompt='Enter bgrflag',
                                            default=bgr_params['bgrflag'],
                                            valid=[0,1,2,3])
        #
        elif ret == 'cnbgr':
            bgr_params['cnbgr'] = get_int(prompt='Enter col nbgr',
                                          default=bgr_params['cnbgr'],
                                          min=0)
        elif ret == 'cwidth':
            # could use flt??
            bgr_params['cwidth'] = get_int(prompt='Enter col width',
                                           default=bgr_params['cwidth'],
                                           min=0)
        elif ret == 'cpow':
            bgr_params['cpow'] = get_flt(prompt='Enter col pow',
                                         default=bgr_params['cpow'],
                                         min=0.)
        elif ret == 'ctan':
            bgr_params['ctan'] = get_tf(prompt='Enter col tan flag',
                                        default=bgr_params['ctan'])
        #
        elif ret == 'rnbgr':
            bgr_params['rnbgr'] = get_int(prompt='Enter row nbgr',
                                          default=bgr_params['rnbgr'],
                                          min=0)
        elif ret == 'rwidth':
            # could use flt??
            bgr_params['rwidth'] = get_int(prompt='Enter row width',
                                           default=bgr_params['rwidth'],
                                           min=0)
        elif ret == 'rpow':
            bgr_params['rpow'] = get_flt(prompt='Enter row pow',
                                         default=bgr_params['rpow'],
                                         min=0.)
        elif ret == 'rtan':
            bgr_params['rtan'] = get_tf(prompt='Enter row tan flag',
                                        default=bgr_params['rtan'])
        elif ret == 'info':
            show_more(BGR_INFO)
            get_yn(prompt="Continue",default='y')
    return bgr_params
    

################################################################################

