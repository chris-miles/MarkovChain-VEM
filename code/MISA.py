import gillespy2
import numpy

def create_MISA_model(end_time=1000,nSamples=100,parameter_values=None):
    model = gillespy2.Model(name="MISA")
    A00 = gillespy2.Species(name="A00", initial_value=1,mode='discrete' )   
    A01 = gillespy2.Species(name="A01", initial_value=0,mode='discrete' )   
    A10 = gillespy2.Species(name="A10", initial_value=0,mode='discrete' )   
    A11 = gillespy2.Species(name="A11", initial_value=0,mode='discrete' )   
    B00 = gillespy2.Species(name="B00", initial_value=1,mode='discrete' )   
    B01 = gillespy2.Species(name="B01", initial_value=0,mode='discrete' )   
    B10 = gillespy2.Species(name="B10", initial_value=0,mode='discrete' )   
    B11 = gillespy2.Species(name="B11", initial_value=0,mode='discrete' )


    a = gillespy2.Species(name="a", initial_value=50,mode='discrete' )
    b = gillespy2.Species(name="b", initial_value=50,mode='discrete' )

    model.add_species([A00,A01,A10,A11,B00,B01,B10,B11,a,b])

    g0 = gillespy2.Parameter(name="g0", expression=10.0)
    g1 = gillespy2.Parameter(name="g1", expression=100.0)
    k = gillespy2.Parameter(name="k", expression=1.0)
    hr = gillespy2.Parameter(name="hr", expression=0.0001)
    fr = gillespy2.Parameter(name="fr", expression=9e-3)
    ha = gillespy2.Parameter(name="ha", expression=0.1)
    fa = gillespy2.Parameter(name="fa", expression=1.0)

    model.add_parameter([g0,g1,k,hr,fr,ha,fa])  
   # Define Reactions
    reaction1 = gillespy2.Reaction(
        name="reaction1", reactants={'A00': 1}, products={'A00': 1, 'a': 1},    rate='g0')
    
    reaction2 = gillespy2.Reaction(
        name="reaction2", reactants={'A01': 1}, products={'A01': 1, 'a': 1},    rate='g0')     

    reaction3 = gillespy2.Reaction(
        name="reaction3", reactants={'A10': 1}, products={'A10': 1, 'a': 1},    rate='g1')
    
    reaction4 = gillespy2.Reaction(
        name="reaction4", reactants={'A11': 1}, products={'A11': 1, 'a': 1},    rate='g0')  

    reaction5 = gillespy2.Reaction(
        name="reaction5", reactants={'B00': 1}, products={'B00': 1, 'b': 1},    rate='g0')
    
    reaction6 = gillespy2.Reaction(
        name="reaction6", reactants={'B01': 1}, products={'B01': 1, 'b': 1},    rate='g0')     

    reaction7 = gillespy2.Reaction(
        name="reaction7", reactants={'B10': 1}, products={'B10': 1, 'b': 1},    rate='g1')
    
    reaction8 = gillespy2.Reaction(
        name="reaction8", reactants={'B11': 1}, products={'B11': 1, 'b': 1},    rate='g0')         
    
    reaction9 = gillespy2.Reaction(
        name="reaction9", reactants={'a': 1}, products={},    rate='k')         
       
    reaction10 = gillespy2.Reaction(
        name="reaction10", reactants={'b': 1}, products={},    rate='k') 

    # r reactions
    reaction11 = gillespy2.Reaction(
        name="reaction11", reactants={'A00': 1, 'b': 2}, products={'A01':1},  propensity_function="hr * A00 * b*(b-1) / 2.0") 
    
    reaction12 = gillespy2.Reaction(
        name="reaction12", reactants={'A01':1}, products={'A00': 1, 'b': 2},    propensity_function='fr * A01') 

    reaction13 = gillespy2.Reaction(
        name="reaction13", reactants={'A10': 1, 'b': 2}, products={'A11':1},    propensity_function='hr * A10 * b*(b-1) / 2.0') 
    
    reaction14 = gillespy2.Reaction(
        name="reaction14", reactants={'A11':1}, products={'A10': 1, 'b': 2},    propensity_function='fr * A11') 
    
    reaction15 = gillespy2.Reaction(
        name="reaction15", reactants={'B00': 1, 'a': 2}, products={'B01':1},    propensity_function='hr * B00 * a*(a-1) / 2.0') 
    
    reaction16 = gillespy2.Reaction(
        name="reaction16", reactants={'B01':1}, products={'B00': 1, 'a': 2},    propensity_function='fr * B01') 

    reaction17 = gillespy2.Reaction(
        name="reaction17", reactants={'B10': 1, 'a': 2}, products={'B11':1},    propensity_function='hr * B10*a*(a-1) / 2.0') 
    
    reaction18 = gillespy2.Reaction(
        name="reaction18", reactants={'B11':1}, products={'B10': 1, 'a': 2},    propensity_function='fr * B11') 
    
    # a reactions

    reaction19 = gillespy2.Reaction(
        name="reaction19", reactants={'A00': 1, 'a': 2}, products={'A10':1},    propensity_function='ha * A00 * a*(a-1) / 2.0') 
    
    reaction20 = gillespy2.Reaction(
        name="reaction20", reactants={'A10':1}, products={'A00': 1, 'a': 2},    propensity_function='fa * A10') 

    reaction21 = gillespy2.Reaction(
        name="reaction21", reactants={'A01': 1, 'a': 2}, products={'A11':1},    propensity_function='ha * A01 * a*(a-1) / 2.0') 
    
    reaction22 = gillespy2.Reaction(
        name="reaction22", reactants={'A11':1}, products={'A01': 1, 'a': 2},    propensity_function='fa * A11') 
    
    reaction23 = gillespy2.Reaction(
        name="reaction23", reactants={'B00': 1, 'b': 2}, products={'B10':1},    propensity_function='ha * B00 * b*(b-1) / 2.0') 
    
    reaction24 = gillespy2.Reaction(
        name="reaction24", reactants={'B10':1}, products={'B00': 1, 'b': 2},    propensity_function='fa * B10') 

    reaction25 = gillespy2.Reaction(
        name="reaction25", reactants={'B01': 1, 'b': 2}, products={'B11':1},    propensity_function='ha * B01 * b*(b-1) / 2.0') 
    
    reaction26 = gillespy2.Reaction(
        name="reaction26", reactants={'B11':1}, products={'B01': 1, 'b': 2},    propensity_function='fa * B11')       
   
    model.add_reaction([reaction1,reaction2,reaction3,reaction4,reaction5,reaction6,reaction7,reaction8,reaction9,reaction10,reaction11,reaction12,reaction13,reaction14,reaction15,reaction16,reaction17,reaction18,reaction19,reaction20,reaction21,reaction22,reaction23,reaction24,reaction25,reaction26])

    tspan = gillespy2.TimeSpan.linspace(t=end_time,num_points=int(nSamples))
    
    # Set Model Timespan
    model.timespan(tspan)
    return model


def create_MISA_model_numpy(end_time,nSamples,frval,astart,bstart):

    model = gillespy2.Model(name="MISA")
    A00 = gillespy2.Species(name="A00", initial_value=1,mode='discrete' )   
    A01 = gillespy2.Species(name="A01", initial_value=0,mode='discrete' )   
    A10 = gillespy2.Species(name="A10", initial_value=0,mode='discrete' )   
    A11 = gillespy2.Species(name="A11", initial_value=0,mode='discrete' )   
    B00 = gillespy2.Species(name="B00", initial_value=1,mode='discrete' )   
    B01 = gillespy2.Species(name="B01", initial_value=0,mode='discrete' )   
    B10 = gillespy2.Species(name="B10", initial_value=0,mode='discrete' )   
    B11 = gillespy2.Species(name="B11", initial_value=0,mode='discrete' )
    #numpy.random.randint(0,numpy.round(100.0)
    a = gillespy2.Species(name="a", initial_value=astart,mode='discrete' )
    b = gillespy2.Species(name="b", initial_value=bstart,mode='discrete' )

    model.add_species([A00,A01,A10,A11,B00,B01,B10,B11,a,b])


    g0 = gillespy2.Parameter(name="g0", expression=10.0)
    g1 = gillespy2.Parameter(name="g1", expression=100.0)
    k = gillespy2.Parameter(name="k", expression=1.0)
    hr = gillespy2.Parameter(name="hr", expression=0.0001)
    fr = gillespy2.Parameter(name="fr", expression=frval)
    ha = gillespy2.Parameter(name="ha", expression=0.1)
    fa = gillespy2.Parameter(name="fa", expression=1.0)


    model.add_parameter([g0,g1,k,hr,fr,ha,fa])  
   # Define Reactions
    reaction1 = gillespy2.Reaction(
        name="reaction1", reactants={'A00': 1}, products={'A00': 1, 'a': 1},    rate='g0')
    
    reaction2 = gillespy2.Reaction(
        name="reaction2", reactants={'A01': 1}, products={'A01': 1, 'a': 1},    rate='g0')     

    reaction3 = gillespy2.Reaction(
        name="reaction3", reactants={'A10': 1}, products={'A10': 1, 'a': 1},    rate='g1')
    
    reaction4 = gillespy2.Reaction(
        name="reaction4", reactants={'A11': 1}, products={'A11': 1, 'a': 1},    rate='g0')  

    reaction5 = gillespy2.Reaction(
        name="reaction5", reactants={'B00': 1}, products={'B00': 1, 'b': 1},    rate='g0')
    
    reaction6 = gillespy2.Reaction(
        name="reaction6", reactants={'B01': 1}, products={'B01': 1, 'b': 1},    rate='g0')     

    reaction7 = gillespy2.Reaction(
        name="reaction7", reactants={'B10': 1}, products={'B10': 1, 'b': 1},    rate='g1')
    
    reaction8 = gillespy2.Reaction(
        name="reaction8", reactants={'B11': 1}, products={'B11': 1, 'b': 1},    rate='g0')         
    
    reaction9 = gillespy2.Reaction(
        name="reaction9", reactants={'a': 1}, products={},    rate='k')         
       
    reaction10 = gillespy2.Reaction(
        name="reaction10", reactants={'b': 1}, products={},    rate='k') 

    # r reactions
    reaction11 = gillespy2.Reaction(
        name="reaction11", reactants={'A00': 1, 'b': 2}, products={'A01':1},  propensity_function="hr * A00 * b*(b-1) / 2") 
    
    reaction12 = gillespy2.Reaction(
        name="reaction12", reactants={'A01':1}, products={'A00': 1, 'b': 2},    propensity_function='fr * A01') 

    reaction13 = gillespy2.Reaction(
        name="reaction13", reactants={'A10': 1, 'b': 2}, products={'A11':1},    propensity_function='hr * A10 * b*(b-1) / 2.0') 
    
    reaction14 = gillespy2.Reaction(
        name="reaction14", reactants={'A11':1}, products={'A10': 1, 'b': 2},    propensity_function='fr * A11') 
    
    reaction15 = gillespy2.Reaction(
        name="reaction15", reactants={'B00': 1, 'a': 2}, products={'B01':1},    propensity_function='hr * B00 * a*(a-1) / 2.0') 
    
    reaction16 = gillespy2.Reaction(
        name="reaction16", reactants={'B01':1}, products={'B00': 1, 'a': 2},    propensity_function='fr * B01') 

    reaction17 = gillespy2.Reaction(
        name="reaction17", reactants={'B10': 1, 'a': 2}, products={'B11':1},    propensity_function='hr * B10*a*(a-1) / 2.0') 
    
    reaction18 = gillespy2.Reaction(
        name="reaction18", reactants={'B11':1}, products={'B10': 1, 'a': 2},    propensity_function='fr * B11') 
    
    # a reactions

    reaction19 = gillespy2.Reaction(
        name="reaction19", reactants={'A00': 1, 'a': 2}, products={'A10':1},    propensity_function='ha * A00 * a*(a-1) / 2.0') 
    
    reaction20 = gillespy2.Reaction(
        name="reaction20", reactants={'A10':1}, products={'A00': 1, 'a': 2},    propensity_function='fa * A10') 

    reaction21 = gillespy2.Reaction(
        name="reaction21", reactants={'A01': 1, 'a': 2}, products={'A11':1},    propensity_function='ha * A01 * a*(a-1) / 2.0') 
    
    reaction22 = gillespy2.Reaction(
        name="reaction22", reactants={'A11':1}, products={'A01': 1, 'a': 2},    propensity_function='fa * A11') 
    
    reaction23 = gillespy2.Reaction(
        name="reaction23", reactants={'B00': 1, 'b': 2}, products={'B10':1},    propensity_function='ha * B00 * b*(b-1) / 2.0') 
    
    reaction24 = gillespy2.Reaction(
        name="reaction24", reactants={'B10':1}, products={'B00': 1, 'b': 2},    propensity_function='fa * B10') 

    reaction25 = gillespy2.Reaction(
        name="reaction25", reactants={'B01': 1, 'b': 2}, products={'B11':1},    propensity_function='ha * B01 * b*(b-1) / 2.0') 
    
    reaction26 = gillespy2.Reaction(
        name="reaction26", reactants={'B11':1}, products={'B01': 1, 'b': 2},    propensity_function='fa * B11')       
   
    model.add_reaction([reaction1,reaction2,reaction3,reaction4,reaction5,reaction6,reaction7,reaction8,reaction9,reaction10,reaction11,reaction12,reaction13,reaction14,reaction15,reaction16,reaction17,reaction18,reaction19,reaction20,reaction21,reaction22,reaction23,reaction24,reaction25,reaction26])

    tspan = gillespy2.TimeSpan.linspace(t=end_time,num_points=int(round(nSamples)))
    
    # Set Model Timespan
    model.timespan(tspan)
    return model

