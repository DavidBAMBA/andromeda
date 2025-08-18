"""
===============================================================================
Thind accretion disk with a simple linear model of spectrum
===============================================================================
@author: Alexis Larrañaga - 2023
===============================================================================
"""

class structure:
    def __init__(self, blackhole, R_min=False , R_max=20., corotating=True):
        if R_min:
            self.in_edge = R_min
        else:
            if corotating:
                self.in_edge = blackhole.ISCOco
            else:
                self.in_edge = blackhole.ISCOcounter
        
        self.out_edge = R_max

    def intensity(self, r):
        '''
        Linear model of the spectrum of the accretion disk
        '''
        m = (1.-0.)/(self.in_edge - self.out_edge)
        I = m * (r - self.out_edge)
        if r>self.in_edge and r<self.out_edge:
            return I
        else:
            return 0.



###############################################################################

if __name__ == '__main__':
    print('')
    print('THIS IS A MODULE DEFINING ONLY A PART OF THE COMPLETE CODE.')
    print('YOU NEED TO RUN THE main.py FILE TO GENERATE THE IMAGE')
    print('')
