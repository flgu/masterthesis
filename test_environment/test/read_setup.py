# -*- coding: utf-8 -*-
class setup():
    
    def __init__( self, path_2_file ):
        
        # define entity list
        entity_list = ["I", "L", "c0", "D0", "epsilon_m", "N", "DC", "DA", "epsilon", "Dt", "phi0", "T"]
    
        with open( path_2_file, mode = "r" ) as file_id:
    
            for line in file_id:
                b = line.strip().split("=")
                
                if b[0].strip() in entity_list:
                    
                    # check if attribute is set
                    if b.__len__() != 2:
                        raise ValueError("Setup parameter " + b[0] + " is not set")
                    else:
                        if b[0].strip() in ["I", "N"]:
                            setattr(self, b[0].strip(), int(b[1].strip()))
                        else:
                            setattr(self, b[0].strip(), float(b[1].strip()))

        self.T0 = self.L ** 2 / self.D0
        self.f0 = self.L * self.c0 / self.T0 # calculate reference flux density

