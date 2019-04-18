import math, os
import pandas as pd

# Pulled right out of the Python notebook
# Does data cleaning and conversion on data/opendata_projects000.gz
# Also removes columns that are currently not used - will have to be edited as we decide more columns are useful
# params:
#   output_filename: If none, returns a pandas DataFrame. Otherwise, outputs the cleaned data to the provided filename as a csv.
def load_projects(data_path, output_filename=None):
    # from DonorsChoose Open Data
    projects_df = pd.read_csv(data_path + 'opendata_projects000.gz', escapechar='\\', names=['_projectid', '_teacher_acctid', '_schoolid', 'school_ncesid', 'school_latitude', 'school_longitude', 'school_city', 'school_state', 'school_zip', 'school_metro', 'school_district', 'school_county', 'school_charter', 'school_magnet', 'school_year_round', 'school_nlns', 'school_kipp', 'school_charter_ready_promise', 'teacher_prefix', 'teacher_teach_for_america', 'teacher_ny_teaching_fellow', 'primary_focus_subject', 'primary_focus_area' ,'secondary_focus_subject', 'secondary_focus_area', 'resource_type', 'poverty_level', 'grade_level', 'vendor_shipping_charges', 'sales_tax', 'payment_processing_charges', 'fulfillment_labor_materials', 'total_price_excluding_optional_support', 'total_price_including_optional_support', 'students_reached', 'total_donations', 'num_donors', 'eligible_double_your_impact_match', 'eligible_almost_home_match', 'funding_status', 'date_posted', 'date_completed', 'date_thank_you_packet_mailed', 'date_expiration'])

    ####################
    # Fix Entry Errors #
    ####################

    # Missing Capitalization - affects 3 rows
    projects_df.loc[projects_df.school_state == "La", "school_state"] = "LA"

    # Missing students_reached values - affects 155 rows
    projects_df.dropna(subset=['students_reached'], inplace=True)

    # ignore all projects that aren't labeled "completed" or "expired"
    projects_df = projects_df[projects_df.funding_status.isin(["completed", "expired"])]

    ######################################################
    # Recast Categorical, Boolean and DateTime Variables #
    ######################################################

    # mapping for matching month to date. Just for ease of reading
    month_map = {
        1:"January",
        2:"February",
        3:"March",
        4:"April",
        5:"May",
        6:"June",
        7:"July",
        8:"August",
        9:"September",
        10:"October",
        11:"November",
        12:"December"
    }

    # Categorical Variables
    projects_df.school_state = projects_df.school_state.astype("category")
    projects_df.school_metro = projects_df.school_metro.astype("category")
    projects_df.teacher_prefix = projects_df.teacher_prefix.astype("category")
    projects_df.resource_type = projects_df.resource_type.astype("category")
    projects_df.poverty_level = projects_df.poverty_level.astype("category")
    projects_df.grade_level = projects_df.grade_level.astype("category")
    projects_df.funding_status = projects_df.funding_status.astype("category")
    projects_df.primary_focus_area = projects_df.primary_focus_area.astype("category")
    projects_df.primary_focus_subject = projects_df.primary_focus_subject.astype("category")
    projects_df.secondary_focus_area = projects_df.secondary_focus_area.astype("category")
    projects_df.secondary_focus_subject = projects_df.secondary_focus_subject.astype("category")

    # Boolean Variables (Cast so that False=0 and True=1)
    projects_df.school_charter = projects_df.school_charter.map({"f":0, "t":1})
    projects_df.school_magnet = projects_df.school_magnet.map({"f":0, "t":1})
    projects_df.school_year_round = projects_df.school_year_round.map({"f":0, "t":1})
    projects_df.school_nlns = projects_df.school_nlns.map({"f":0, "t":1})
    projects_df.school_kipp = projects_df.school_kipp.map({"f":0, "t":1})
    projects_df.school_charter_ready_promise = projects_df.school_charter_ready_promise.map({"f":0, "t":1})
    projects_df.teacher_teach_for_america = projects_df.teacher_teach_for_america.map({"f":0, "t":1})
    projects_df.teacher_ny_teaching_fellow = projects_df.teacher_ny_teaching_fellow.map({"f":0, "t":1})
    projects_df.eligible_double_your_impact_match = projects_df.eligible_double_your_impact_match.map({"f":0, "t":1})
    projects_df.eligible_almost_home_match = projects_df.eligible_almost_home_match.map({"f":0, "t":1})

    # Datetime Variable (Only date posted - create a few categorical colums from that)
    projects_df.date_posted = pd.to_datetime(projects_df.date_posted)
    projects_df['month_posted'] = projects_df.date_posted.map(lambda x: x.month)
    projects_df['week_of_year_posted'] = projects_df.date_posted.map(lambda x: x.weekofyear)

    projects_df.month_posted = projects_df.month_posted.map(month_map).astype("category")

    # one hot encoding for categorical variables
    projects_df = projects_df.merge(pd.get_dummies(projects_df.school_metro), left_index=True, right_index=True)
    projects_df = projects_df.merge(pd.get_dummies(projects_df.teacher_prefix), left_index=True, right_index=True)
    projects_df = projects_df.merge(pd.get_dummies(projects_df.month_posted), left_index=True, right_index=True)
    projects_df = projects_df.merge(pd.get_dummies(projects_df.poverty_level), left_index=True, right_index=True)
    projects_df = projects_df.merge(pd.get_dummies(projects_df.grade_level), left_index=True, right_index=True)

    # Avoid issue with identical category names by adding a prefix to the dummy columns before merging them
    dummies = pd.get_dummies(projects_df.resource_type)
    dummies.columns = ["Resource_" + col for col in projects_df.resource_type.cat.categories]
    projects_df = projects_df.merge(dummies, left_index=True, right_index=True)

    # we want to combine the primary and secondary area and subjects in the one hot columns
    # add the columns together, but in some instances the user submitted the same category
    # for both primary and secondary, so we want to make sure the columns are only 0 and 1
    # also ditto on the column names like for resources
    dummies = pd.get_dummies(projects_df.primary_focus_area)+pd.get_dummies(projects_df.secondary_focus_area)
    dummies.columns = ["Focus_Area_" + col for col in projects_df.primary_focus_area.cat.categories]
    projects_df = projects_df.merge(dummies, left_index=True, right_index=True)
    for col in dummies.columns:
        projects_df[col] = projects_df[col].apply(lambda x: 1 if x == 2 else x)

    dummies = pd.get_dummies(projects_df.primary_focus_subject)+pd.get_dummies(projects_df.secondary_focus_subject)
    dummies.columns = ["Focus_Subject_" + col for col in projects_df.primary_focus_subject.cat.categories]
    projects_df = projects_df.merge(dummies, left_index=True, right_index=True)
    for col in dummies.columns:
        projects_df[col] = projects_df[col].apply(lambda x: 1 if x == 2 else x)

    # map ordinal variables
    projects_df['poverty_level_code'] = projects_df.poverty_level.map({"highest poverty":3, "high poverty":2, "moderate poverty":1, "low poverty":0})
    projects_df['grade_level_code'] = projects_df.grade_level.map({"Grades 9-12":4, "Grades 6-8":3, "Grades 3-5":2, "Grades PreK-2":1})
    projects_df['grade_level_code'] = projects_df['grade_level_code'].fillna(0)

    ###################
    # Columns To Drop #
    ###################

    # ids aren't needed by the classifier
    # projects_df.drop(columns=['_teacher_acctid', '_schoolid'], inplace=True)

    # no geospatial analysis at the moment
    # projects_df.drop(columns=['school_latitude', 'school_longitude'], inplace=True)

    # leaving out individual school details at the moment
    # projects_df.drop(columns=['school_city', 'school_state', 'school_zip', 'school_district', 'school_county'], inplace=True)

    # temporal fields aren't needed
    projects_df.drop(columns=['date_completed','date_thank_you_packet_mailed','date_expiration'], inplace=True)

    # these fields are all correlated with the total price
    projects_df.drop(columns=['vendor_shipping_charges', 'sales_tax','payment_processing_charges','fulfillment_labor_materials', 'total_price_excluding_optional_support'], inplace=True)

    # Not enough data in this column to make a prediction
    projects_df.drop(columns=['Mr. & Mrs.'], inplace=True)

    # These fields violate >t0 for predictions
    projects_df.drop(columns=['total_donations', 'num_donors', 'eligible_double_your_impact_match', 'eligible_almost_home_match'], inplace=True)

    #######################
    # Feature Engineering #
    #######################

    projects_df['LOG_total_price'] = projects_df.total_price_including_optional_support.apply(lambda x: math.log(x) if x > 0 else 0)
    projects_df['LOG_students_reached'] = projects_df.students_reached.apply(lambda x: math.log(x) if x > 0 else 0)

    #################
    # Export to CSV #
    #################

    print("Dataset sizes:")
    print("projects_df: {0:,} rows, {1} variables".format(*projects_df.shape))

    if output_filename == None:
        return projects_df
    else:
        projects_df.to_csv(output_filename)
