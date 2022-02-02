import pandas as pd
import config



accounts_df=pd.read_csv(config.twitter_acc_dir, header=0)
lista_cuentas_twitter= accounts_df.account_name.values.tolist()

