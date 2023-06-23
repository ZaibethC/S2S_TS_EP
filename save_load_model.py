import joblib

SAVE_MODEL_DIRECTORY = "saved_models/"

def get_exp_filename(exp_name, leadtime=None, features_type=None, iterfold=None):

    exp_filename = exp_name

    if leadtime is not None:
        exp_filename = exp_filename + '_leadtime_' + str(leadtime)
    if features_type is not None:
        exp_filename = exp_filename + '_featurestype_' + features_type
    if iterfold is not None:
        exp_filename = exp_filename + "_iterfold_" + str(iterfold)

    return exp_filename


def save_model(exp_filename, rf):

    model_filename = SAVE_MODEL_DIRECTORY + exp_filename + '.jbl.lzma'
    with open(model_filename, "wb") as f:
        joblib.dump(rf, model_filename)

    return model_filename


def load_model(exp_filename):

    model_filename = exp_filename + '.jbl.lzma'

    return joblib.load(model_filename)
