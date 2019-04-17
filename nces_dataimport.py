import pandas as pd
import numpy as np
import pickle, os, zipfile, urllib, re, struct
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

NCES_base_url = 'https://nces.ed.gov/ccd/Data/{}/{}.{}'

####################
# Helper Functions #
####################

# Check if the file exists in the datapath, and if not, downloads it
# params:
#     filename: file to search for
#     file_format: the format of the file (zip or txt)
# returns:
#     None
def file_check(filename, file_format="zip"):
    file_path = "{}{}.{}".format(data_path, filename, file_format)

    if not os.path.isfile(file_path):
        url = NCES_base_url.format(file_format, filename,file_format)
        print("File not found - downloading from", url , '...')
        try:
            urllib.request.urlretrieve(
                url=url,
                filename=file_path
            )
        except Exception as e:
            print(e)
            print(url)

# reads from the given file into a pandas DataFrame and returns it
# params:
#     file_name: file to download/search for
#     col_mod (optional): string to remove from column names (some files append the year to headers, e.g. "WHALM09" -> "WHALM")
# returns:
#     DataFrame
def get_and_load_zip(file_name, col_mod=None):
    # make sure we download this file if we don't already have it
    file_check(file_name+"_txt")

    headers = []
    data = []
    with zipfile.ZipFile(data_path + '{}_txt.zip'.format(file_name), 'r') as archive:
        with archive.open("{}.txt".format(file_name)) as f:
            count = 0
            for line in f.readlines():
                if count == 0:
                    headers = [x.strip() for x in line.decode("ISO-8859-1").split('\t')]
                else:
                    data.append([x.strip() for x in line.decode("ISO-8859-1").split('\t')])
                count +=1

    df = pd.DataFrame(data)

    if file_name == "sc111a_supp":
        # this one file is screwy - drop the problematic rows and the extra column
        df.drop([3039, 3040], inplace=True)
        df.drop(columns=[322], inplace=True)

    if col_mod is None:
        df.columns = headers
    else:
#         df.columns = [x.replace(col_mod, '') for x in headers]
        df.columns = [re.sub(r"{}\b".format(col_mod),'',x) for x in headers]
    return df

# reads from the given file into a pandas DataFrame and returns it
# params:
#     layout_file: the name of the layout file to get column headers / layout information from
#     dat_files: a list of the .dat filenames to combine together
#     col_mod (optional): string to remove from column names (some files append the year to headers, e.g. "WHALM09" -> "WHALM")
#     file_type: the format of the file (zip or txt) WITHIN the zip
# returns:
#     DataFrame
def get_and_load_dat(layout_file, dat_files, col_mod, file_type="dat"):
    # check that we've downloaded the layout file
    file_check(layout_file, file_format="txt")

    # parse the data file layout file
    layout = []
    with open(data_path+layout_file+".txt", encoding="ISO-8859-1") as f:
        for line in f.readlines():
            splits = line.split()
            # we'll find the layout by looking for lines that split into something like "[NCESSCH,1,12]"
            if len(splits) > 3 and "+" not in splits[0] and tryParseInt(splits[1]) and tryParseInt(splits[2]):
                col_name = splits[0]
                col_width = int(re.sub(r'[^0-9]', '', splits[3])) # make sure we only grab numeric values.
                col_type = splits[4]
                layout.append((col_name, col_width))

    # treat all fields as strings - it doesn't matter to us for the moment
    # (we can get this if it turns out we need it later - "col_type" from above)
    fieldstruct = struct.Struct(' '.join([str(x[1])+"s" for x in layout]))

    # similar to get_and_load_zip
    dat_file_dfs = []
    error_count = 0
    for dat_file in dat_files:
        file_check(dat_file.lower()) # the zip is always lower, the files sometimes start with upper. v0v

        data = []
        with zipfile.ZipFile(data_path + '{}.zip'.format(dat_file.lower()), 'r') as archive:
            with archive.open("{}.{}".format(dat_file.split('_')[0], file_type)) as f: # TODO: this is bad.
                for line in f.readlines():
                    try:
                        data.append([d.decode().strip() for d in fieldstruct.unpack_from(line)])
                    except:
                        # keep track of the number of times we fail to parse a line.
                        error_count+=1
        dat_file_dfs.append(pd.DataFrame(data))

    print("unable to parse {} rows...".format(error_count))

    combined_dfs = pd.concat(dat_file_dfs)
    # strip the date from column headers
    combined_dfs.columns = [re.sub(r"{}\b".format(col_mod),'',x[0]) for x in layout]

    return combined_dfs

# Trims a dataframe by looking at the projects df to see what ids appear in a given year
# also creates a simple column with just the year
# params:
#     layout_file: the name of the layout file to get column headers / layout information from
#     dat_files: a list of the .dat filenames to combine together
# returns:
#     DataFrame
def trim_by_ID_and_year(df, id_col, year):
    valid_ids = set(projects_df[projects_df['year']==year]['school_ncesid'].values)
    print("Found {} ids in projects_df in {}...".format(len(valid_ids), year))
    df = df[df[id_col].isin(valid_ids)]

    # As long as we're here with the year and all, lets make a year column
    df['year'] = year


    print("Found {} matching rows ({:2.2f}%)".format(df.shape[0], (df.shape[0]/len(valid_ids)*100)))
    return df


# returns True if the string can be parsed into an int
# params:
#     s: the string to check
# returns:
#     Boolean
def tryParseInt(s):
    try:
        int(s)
        return True
    except:
        return False

# pulls data from the NCES site (https://nces.ed.gov/ccd/Data), parses and saves it to local files
# params:
#     data_path: the file location to save all data files downloaded/created
# returns:
#     None. Saves a csv file to disk with all imported data.
def do_import(data_path, output_filename="NCES_Data.csv"):
    # To keep the size of the NCES data down, we can make sure we only bother keeping data relevant to our dataset
    # load in the projects dataset and create a dataframe with only project id, NCESID and the year posted
    projects_df = pd.read_csv(data_path + 'opendata_projects000.gz', escapechar='\\', names=['_projectid', '_teacher_acctid', '_schoolid', 'school_ncesid', 'school_latitude', 'school_longitude', 'school_city', 'school_state', 'school_zip', 'school_metro', 'school_district', 'school_county', 'school_charter', 'school_magnet', 'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise', 'teacher_prefix', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow', 'primary_focus_subject', 'primary_focus_area' ,'secondary_focus_subject', 'secondary_focus_area', 'resource_type', 'poverty_level', 'grade_level', 'vendor_shipping_charges', 'sales_tax', 'payment_processing_charges', 'fulfillment_labor_materials', 'total_price_excluding_optional_support', 'total_price_including_optional_support', 'students_reached', 'total_donations', 'num_donors', 'eligible_double_your_impact_match', 'eligible_almost_home_match', 'funding_status', 'date_posted', 'date_completed', 'date_thank_you_packet_mailed', 'date_expiration'])
    projects_df = projects_df[['_projectid', 'school_ncesid', 'date_posted']]

    # drop any rows which are missing either a NCES id or a posted date, since those are useless for this
    projects_df.dropna(inplace=True)

    # convert the two columns we're intersted in to the proper format
    projects_df.school_ncesid = projects_df.school_ncesid.astype("int64")
    projects_df.school_ncesid = projects_df.school_ncesid.apply(lambda x: str(x).zfill(12))
    projects_df.date_posted = pd.to_datetime(projects_df.date_posted)

    # derive the year from the data posted column
    # we're calling all proposals prior to june part of the previous school year
    projects_df['year'] = projects_df.date_posted.apply(lambda x: x.year if x.month > 6 else x.year-1)

    #######################
    # NCES 2014-2015 Data #
    #######################
    # the 14-15 data is split into 5 different files
    ccd1415_directory = get_and_load_zip('ccd_sch_029_1415_w_0216601a')
    ccd1415_membership = get_and_load_zip('ccd_sch_052_1415_w_0216161a')
    ccd1415_staff = get_and_load_zip('ccd_sch_059_1415_w_0216161a')
    ccd1415_lunch = get_and_load_zip('ccd_sch_033_1415_w_0216161a')
    ccd1415_schoolchar = get_and_load_zip('ccd_sch_129_1415_w_0216161a')

    # get a list of column names in common so we can drop them as we merge
    common_columns = list(set(list(ccd1415_directory.columns)).intersection(list(ccd1415_membership.columns)))
    common_columns.remove('NCESSCH') # keep this value - it's what we'll match on

    # Merge all of these dataframes together matching on the NCES ID
    ccd1415 = ccd1415_directory.merge(ccd1415_membership.drop(common_columns, axis=1), on="NCESSCH")
    ccd1415 = ccd1415.merge(ccd1415_staff.drop(common_columns, axis=1), on="NCESSCH")
    ccd1415 = ccd1415.merge(ccd1415_lunch.drop(common_columns, axis=1), on="NCESSCH")
    ccd1415 = ccd1415.merge(ccd1415_schoolchar.drop(common_columns, axis=1), on="NCESSCH")

    # trim the dataframe to only rows we'll use
    ccd1415 = trim_by_ID_and_year(ccd1415, 'NCESSCH', 2014)

    #######################
    # NCES 2013-2014 Data #
    #######################
    ccd1314 = get_and_load_zip('sc132a')
    ccd1314 = trim_by_ID_and_year(ccd1314, 'NCESSCH', 2013)

    #######################
    # NCES 2013-2014 Data #
    #######################
    ccd1314 = get_and_load_zip('sc132a')
    ccd1314 = trim_by_ID_and_year(ccd1314, 'NCESSCH', 2013)

    #######################
    # NCES 2012-2013 Data #
    #######################
    ccd1213 = get_and_load_zip('sc122a')
    ccd1213 = trim_by_ID_and_year(ccd1213, 'NCESSCH', 2012)

    #######################
    # NCES 2011-2012 Data #
    #######################
    ccd1112 = get_and_load_zip('sc111a_supp')
    ccd1112 = trim_by_ID_and_year(ccd1112, 'NCESSCH', 2011)

    #######################
    # NCES 2010-2011 Data #
    #######################
    ccd1011 = get_and_load_zip('sc102a')
    ccd1011 = trim_by_ID_and_year(ccd1011, 'NCESSCH', 2010)

    #######################
    # NCES 2009-2010 Data #
    #######################
    ccd0910 = get_and_load_zip('sc092a', col_mod='09')
    ccd0910 = trim_by_ID_and_year(ccd0910, 'NCESSCH', 2009)

    #######################
    # NCES 2008-2009 Data #
    #######################
    ccd0809 = get_and_load_zip('sc081b', col_mod='08')
    ccd0809 = trim_by_ID_and_year(ccd0809, 'NCESSCH', 2008)

    #######################
    # NCES 2007-2008 Data #
    #######################
    ccd0708 = get_and_load_zip('sc071b', col_mod='07')
    ccd0708 = trim_by_ID_and_year(ccd0708, 'NCESSCH', 2007)

    #######################
    # NCES 2006-2007 Data #
    #######################
    layout_file = "psu061clay"
    dat_files = ["Sc061cai_dat", "Sc061ckn_dat", "Sc061cow_dat"]
    ccd0607 = get_and_load_dat(layout_file, dat_files, '06')
    ccd0607 = trim_by_ID_and_year(ccd0607, 'NCESSCH', 2006)

    #######################
    # NCES 2005-2006 Data #
    #######################
    layout_file = "psu051alay"
    dat_files = ["Sc051aai_dat", "Sc051akn_dat", "Sc051aow_dat"]
    ccd0506 = get_and_load_dat(layout_file, dat_files, '05')
    ccd0506 = trim_by_ID_and_year(ccd0506, 'NCESSCH', 2005)

    #######################
    # NCES 2004-2005 Data #
    #######################
    layout_file = "psu041blay"
    dat_files = ["sc041bai_dat", "sc041bkn_dat", "sc041bow_dat"]
    ccd0405 = get_and_load_dat(layout_file, dat_files, '04')
    ccd0405 = trim_by_ID_and_year(ccd0405, 'NCESSCH', 2004)

    #######################
    # NCES 2003-2004 Data #
    #######################
    layout_file = "psu031alay"
    dat_files = ["sc031aai_dat", "sc031akn_dat", "sc031aow_dat"]
    ccd0304 = get_and_load_dat(layout_file, dat_files, '03', "txt")
    ccd0304 = trim_by_ID_and_year(ccd0304, 'NCESSCH', 2003)

    #######################
    # NCES 2002-203 Data #
    #######################
    layout_file = "psu021alay"
    dat_files = ["Sc021aai_dat", "Sc021akn_dat", "Sc021aow_dat"]
    ccd0203 = get_and_load_dat(layout_file, dat_files, '02', "txt")
    ccd0203 = trim_by_ID_and_year(ccd0203, 'NCESSCH', 2002)

    ###########################
    # Identify Common Headers #
    ###########################

    # merge all of the headers into a common list
    all_headers = list(ccd1415.columns), \
    list(ccd1314.columns),  \
    list(ccd1213.columns),  \
    list(ccd1112.columns),  \
    list(ccd1011.columns),  \
    list(ccd0910.columns),  \
    list(ccd0809.columns),  \
    list(ccd0708.columns),  \
    list(ccd0607.columns),  \
    list(ccd0506.columns),  \
    list(ccd0405.columns),  \
    list(ccd0304.columns),  \
    list(ccd0203.columns),  \

    overall_headers = set(ccd1415.columns)

    # iterate through all headers to find the intersection with each
    for each in all_headers:
        overall_headers = overall_headers.intersection(each)

    overall_headers = list(overall_headers)

    #################################
    # Merge all dataframes together #
    #################################

    # put the columns in a proper order out of the set by looking at the columns
    # of one of the dataframes
    ordered_headers = []
    for col in ccd1415.columns:
        if col in overall_headers:
            ordered_headers.append(col)

    cdd_allyears= pd.concat([
        ccd1415[ordered_headers],
        ccd1314[ordered_headers],
        ccd1213[ordered_headers],
        ccd1112[ordered_headers],
        ccd1011[ordered_headers],
        ccd0910[ordered_headers],
        ccd0809[ordered_headers],
        ccd0708[ordered_headers],
        ccd0607[ordered_headers],
        ccd0506[ordered_headers],
        ccd0405[ordered_headers],
        ccd0304[ordered_headers],
        ccd0203[ordered_headers],
    ])

    # rename the ncesid column to match the data we're pulling in
    projects_df.rename(columns={"school_ncesid":"NCESSCH"}, inplace=True)

    # Combine the CCD and projects dataframes, merging on the year and NCES ID
    projects_df = projects_df.merge(cdd_allyears, on=['year', 'NCESSCH'])

    # Try to convert each of the columns to integers.
    # Some are not integers, and we expect them to fail
    for col in projects_df.columns[24:]:
        try:
            projects_df[col] = projects_df[col].astype("int")
            projects_df[col] = projects_df[col].apply(lambda x: np.nan if x < 0 else x)
        except:
            print("Can't convert to int:",col)

    # save the file to disk
    projects_df.to_csv(data_path+"NCES_Data.csv", index=None)
