# make a unit conversion module in mpcutil (rename to mpc)

def eV_to_angstrom(E):
    const= 12398.4244
    return const / float(E)