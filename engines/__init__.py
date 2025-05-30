from .base_engine import BaseTrainer
from .common_mil import CommonMIL
from .e2e import E2E

def build_engine(args,device):

    _commom_mil = ('rrtmil','abmil','gabmil','clam_sb','clam_mb','transmil','dsmil','dtfd','meanmil','maxmil','vitmil','abmilx')
    
    if args.model in _commom_mil:
        engine = CommonMIL(args)
    elif args.model.startswith('e2e'):
        engine = E2E(args,device)
    else:
        raise NotImplementedError
    trainer = BaseTrainer(engine=engine,args=args)

    if args.datasets.lower().startswith('surv'):
        return trainer.surv_train,trainer.surv_validate
    else:
        return trainer.train,trainer.validate
