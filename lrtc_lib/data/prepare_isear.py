# (c) Copyright IBM Corporation 2020.

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

# Preliminary processing code, used to create the csv files. Requires special installation to open `.mdb` files
try:
    import pandas_access as mdb  # This is a pandas wrapper for `mdbtools`, mdbtools must be installed separately with homebrew (on mac)


    mdb_file = 'isear/isear_databank.mdb'

    df_full = mdb.read_table(mdb_file, 'DATA')
    df = df_full[['Field1', 'SIT']]
    df = df.rename(columns={"Field1": "label", "SIT": "text"})
    df['text'] = df['text'].apply(lambda x: x.replace('á\n', ''))

    df.to_csv('isear/isear_data.csv', index=False)

except Exception as e:
    print('\n********************************************************************************************************\n'
          '****** Loading the ISEAR dataset requires special dependencies.                                *********\n'
          '****** On Mac/Linux, install https://github.com/mdbtools/mdbtools and `pip install pandas_access`, *****\n'
          '****** and then rerun the main script.                                                             *****\n'
          '********************************************************************************************************\n')

