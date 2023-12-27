import candle
import os
import urllib
# from data_utils import candle_data_dict, download_candle_data
from data_utils import Downloader, candle_data_dict

file_path = os.path.dirname(os.path.realpath(__file__))
additional_definitions = [
    {'name': 'model_name',
     'type': str,
     'help': '...'
     },
    {'name': 'cuda_name',
     'type': str,
     'help': '...'
     },
    {'name': 'data_url',
     'type': str,
     'help': '...'
     },
    {'name': 'download_data',
     'type': bool,
     'help': '...'
     },
    {'name': 'data_source',
     'type': str,
     'help': '...'
     },
    {'name': 'data_split_seed',
     'type': int,
     'help': '.....'
     },
    {'name': 'data_split_id',
     'type': int,
     'help': '.....'
     },
    {'name': 'data_version',
     'type': str,
     'help': '.....'
     },
     {'name': 'data_type',
     'type': str,
     'help': '.....'
     } 
]

required = None
CANDLE_DATA_DIR=os.getenv("CANDLE_DATA_DIR")



def download_ccle_data(args):

    # data_path = os.path.join(CANDLE_DATA_DIR, opt['model_name'], 'Data')
    
    data_type = candle_data_dict[args['data_source']]
    data_split_id = opt['data_split_id']
    data_path=os.path.join(CANDLE_DATA_DIR, args['model_name'], 'Data')
    # download_candle_data(data_type=data_type, split_id=split_id, data_dest=data_path)

    dw = Downloader(args)
    dw.download_candle_data(data_type=data_type, split_id=data_split_id, data_dest=data_path)

# def download_csa_data(opt):

#     csa_data_folder = os.path.join(CANDLE_DATA_DIR, opt['model_name'], 'Data', 'csa_data', 'raw_data')
#     splits_dir = os.path.join(csa_data_folder, 'splits') 
#     x_data_dir = os.path.join(csa_data_folder, 'x_data')
#     y_data_dir = os.path.join(csa_data_folder, 'y_data')

#     if not os.path.exists(csa_data_folder):
#         print('creating folder: %s'%csa_data_folder)
#         os.makedirs(csa_data_folder)
#         os.mkdir( splits_dir  )
#         os.mkdir( x_data_dir  )
#         os.mkdir( y_data_dir  )
    

#     for file in ['CCLE_all.txt', 'CCLE_split_0_test.txt', 'CCLE_split_0_train.txt', 'CCLE_split_0_val.txt']:
#         urllib.request.urlretrieve(f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-imp-2023/csa_data/splits/{file}',
#         splits_dir+f'/{file}')

#     for file in ['cancer_mutation_count.txt', 'drug_SMILES.txt','drug_ecfp4_512bit.txt' ]:
#         urllib.request.urlretrieve(f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-imp-2023/csa_data/x_data/{file}',
#         x_data_dir+f'/{file}')

#     for file in ['response.txt']:
#         urllib.request.urlretrieve(f'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-imp-2023/csa_data/y_data/{file}',
#         y_data_dir+f'/{file}')


class GrapgDRP_candle(candle.Benchmark):

        def set_locals(self):
            if required is not None:
                self.required = set(required)
            if additional_definitions is not None:
                self.additional_definitions = additional_definitions


def initialize_parameters():
    """ Initialize the parameters for the GraphDRP benchmark. """
    print("Initializing parameters\n")
    graphdrp_params = GrapgDRP_candle(
                            filepath=file_path,
                            defmodel="graphdrp_model.txt",
                                            # defmodel="graphdrp_model_candle.txt",
                            framework="pytorch",
                            prog="GraphDRP",
                            desc="CANDLE compliant GraphDRP",
                                )
    gParameters = candle.finalize_parameters(graphdrp_params)
    return gParameters




if __name__ == '__main__':

    opt = initialize_parameters()
    download_ccle_data(opt)

    print("Done.")
