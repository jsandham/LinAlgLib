import urllib.request
import tarfile
import os

def download_tar_file(url, filepath):
    print("Downloading ", filepath)
    urllib.request.urlretrieve(url, filepath)
    print("Downloading ", filepath, " succeeded")

def extract_tar_file(filepath):
    print("Extracting ", filepath)
    with tarfile.open(filepath, "r") as tf:
        tf.extractall(path=os.path.dirname(filepath))

def download_and_extract_matrix(url, filepath):
    download_tar_file(url, filepath)
    extract_tar_file(filepath)


# SPD
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/1138_bus.tar.gz", "SPD/ll38_bus.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk01.tar.gz", "SPD/bcsstk01.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstm02.tar.gz", "SPD/bcsstm02.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Pothen/bodyy4.tar.gz", "SPD/bodyy4.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Pothen/bodyy6.tar.gz", "SPD/bodyy6.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Rothberg/cfd2.tar.gz", "SPD/cfd2.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Boeing/crystm02.tar.gz", "SPD/crystm02.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/FIDAP/ex5.tar.gz", "SPD/ex5.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Norris/fv1.tar.gz", "SPD/fv1.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Norris/fv2.tar.gz", "SPD/fv2.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Norris/fv3.tar.gz", "SPD/fv3.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Pothen/mesh1em6.tar.gz", "SPD/mesh1em6.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Pothen/mesh2em5.tar.gz", "SPD/mesh2em5.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Pothen/mesh3em5.tar.gz", "SPD/mesh3em5.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/nos1.tar.gz", "SPD/nos1.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/nos2.tar.gz", "SPD/nos2.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/nos6.tar.gz", "SPD/nos6.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/ACUSIM/Pres_Poisson.tar.gz", "SPD/Pres_Poisson.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/MaxPlanck/shallow_water1.tar.gz", "SPD/shallow_water1.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/MaxPlanck/shallow_water2.tar.gz", "SPD/shallow_water2.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Schmid/thermal1.tar.gz", "SPD/thermal1.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/thermomech_dM.tar.gz", "SPD/thermomech_dM.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/thermomech_TC.tar.gz", "SPD/thermomech_TC.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/thermomech_TK.tar.gz", "SPD/thermomech_TK.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Nasa/nasa1824.tar.gz", "SPD/nasa1824.tar.gz")

# Symmetric
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/ash85.tar.gz", "Symmetric/ash85.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/can_268.tar.gz", "Symmetric/can_268.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/dwt_1242.tar.gz", "Symmetric/dwt_1242.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Grund/meg4.tar.gz", "Symmetric/meg4.tar.gz")

# Unsymmetric
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/pores_1.tar.gz", "Unsymmetric/pores_1.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Bai/ck104.tar.gz", "Unsymmetric/ck104.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/young1c.tar.gz", "Unsymmetric/young1c.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Bai/olm2000.tar.gz", "Unsymmetric/olm2000.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/arc130.tar.gz", "Unsymmetric/arc130.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/fs_680_3.tar.gz", "Unsymmetric/fs_680_3.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/jpwh_991.tar.gz", "Unsymmetric/jpwh_991.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/plskz362.tar.gz", "Unsymmetric/plskz362.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/DRIVCAV/cavity10.tar.gz", "Unsymmetric/cavity10.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/FIDAP/ex35.tar.gz", "Unsymmetric/ex35.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/thermomech_dK.tar.gz", "Unsymmetric/thermomech_dK.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Hamm/scircuit.tar.gz", "Unsymmetric/scircuit.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Mallya/lhr01.tar.gz", "Unsymmetric/lhr01.tar.gz")