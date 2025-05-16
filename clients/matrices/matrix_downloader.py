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


download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/MathWorks/Kuu.tar.gz", "SPD/Kuu.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Oberwolfach/gyro_m.tar.gz", "SPD/gyro_m.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/cvxbqp1.tar.gz", "SPD/cvxbqp1.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Boeing/bcsstk38.tar.gz", "SPD/bcsstk38.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk17.tar.gz", "SPD/bcsstk17.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/wathen100.tar.gz", "SPD/wathen100.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/gridgena.tar.gz", "SPD/gridgena.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/apache1.tar.gz", "SPD/apache1.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/wathen120.tar.gz", "SPD/wathen120.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Boeing/crystm03.tar.gz", "SPD/crystm03.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Mulvey/finan512.tar.gz", "SPD/finan512.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/TKK/cbuckle.tar.gz", "SPD/cbuckle.tar.gz")




download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Cylshell/s1rmq4m1.tar.gz", "SPD/s1rmq4m1.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/AMD/G2_circuit.tar.gz", "SPD/G2_circuit.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Lourakis/bundle1.tar.gz", "SPD/bundle1.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Simon/olafu.tar.gz", "SPD/olafu.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Oberwolfach/gyro.tar.gz", "SPD/gyro.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/UTEP/Dubcova2.tar.gz", "SPD/Dubcova2.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Boeing/msc23052.tar.gz", "SPD/msc23052.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Boeing/bcsstk36.tar.gz", "SPD/bcsstk36.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Castrillon/denormal.tar.gz", "SPD/denormal.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Boeing/msc10848.tar.gz", "SPD/msc10848.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Simon/raefsky4.tar.gz", "SPD/raefsky4.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Um/2cubes_sphere.tar.gz", "SPD/2cubes_sphere.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Cunningham/qa8fm.tar.gz", "SPD/qa8fm.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Rothberg/cfd1.tar.gz", "SPD/cfd1.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/oilpan.tar.gz", "SPD/oilpan.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/vanbody.tar.gz", "SPD/vanbody.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Boeing/ct20stif.tar.gz", "SPD/ct20stif.tar.gz")

download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/DNVS/shipsec8.tar.gz", "SPD/shipsec8.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/DNVS/shipsec1.tar.gz", "SPD/shipsec1.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/UTEP/Dubcova3.tar.gz", "SPD/Dubcova3.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Wissgott/parabolic_fem.tar.gz", "SPD/parabolic_fem.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/s3dkt3m2.tar.gz", "SPD/s3dkt3m2.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/TKK/smt.tar.gz", "SPD/smt.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/DNVS/ship_003.tar.gz", "SPD/ship_003.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/DNVS/ship_001.tar.gz", "SPD/ship_001.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Williams/cant.tar.gz", "SPD/cant.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Um/offshore.tar.gz", "SPD/offshore.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Williams/pdb1HYS.tar.gz", "SPD/pdb1HYS.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/s3dkq4m2.tar.gz", "SPD/s3dkq4m2.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/DNVS/thread.tar.gz", "SPD/thread.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/DNVS/shipsec5.tar.gz", "SPD/shipsec5.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/apache2.tar.gz", "SPD/apache2.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/McRae/ecology2.tar.gz", "SPD/ecology2.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/CEMW/tmt_sym.tar.gz", "SPD/tmt_sym.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Oberwolfach/boneS01.tar.gz", "SPD/boneS01.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Williams/consph.tar.gz", "SPD/consph.tar.gz")

download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/ND/nd6k.tar.gz", "SPD/nd6k.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/bmw7st_1.tar.gz", "SPD/bmw7st_1.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/AMD/G3_circuit.tar.gz", "SPD/G3_circuit.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Schmid/thermal2.tar.gz", "SPD/thermal2.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/DNVS/x104.tar.gz", "SPD/x104.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/DNVS/m_t1.tar.gz", "SPD/m_t1.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/hood.tar.gz", "SPD/hood.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/crankseg_1.tar.gz", "SPD/crankseg_1.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/bmwcra_1.tar.gz", "SPD/bmwcra_1.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Boeing/pwtk.tar.gz", "SPD/pwtk.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/crankseg_2.tar.gz", "SPD/crankseg_2.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/ND/nd12k.tar.gz", "SPD/nd12k.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_3_k101.tar.gz", "SPD/af_3_k101.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_0_k101.tar.gz", "SPD/af_0_k101.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_1_k101.tar.gz", "SPD/af_1_k101.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_2_k101.tar.gz", "SPD/af_2_k101.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_4_k101.tar.gz", "SPD/af_4_k101.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_shell4.tar.gz", "SPD/af_shell4.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_shell3.tar.gz", "SPD/af_shell3.tar.gz")


download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_shell7.tar.gz", "SPD/af_shell7.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_shell8.tar.gz", "SPD/af_shell8.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/INPRO/msdoor.tar.gz", "SPD/msdoor.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Mazaheri/bundle_adj.tar.gz", "SPD/bundle_adj.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Janna/StocF-1465.tar.gz", "SPD/StocF-1465.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Janna/Fault_639.tar.gz", "SPD/Fault_639.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/ND/nd24k.tar.gz", "SPD/nd24k.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/inline_1.tar.gz", "SPD/inline_1.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Janna/PFlow_742.tar.gz", "SPD/PFlow_742.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Janna/Emilia_923.tar.gz", "SPD/Emilia_923.tar.gz")
download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/ldoor.tar.gz", "SPD/ldoor.tar.gz")








# # Symmetric
# download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/ash85.tar.gz", "Symmetric/ash85.tar.gz")
# download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/can_268.tar.gz", "Symmetric/can_268.tar.gz")
# download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/dwt_1242.tar.gz", "Symmetric/dwt_1242.tar.gz")
# download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Grund/meg4.tar.gz", "Symmetric/meg4.tar.gz")

# # Unsymmetric
# download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/pores_1.tar.gz", "Unsymmetric/pores_1.tar.gz")
# download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Bai/ck104.tar.gz", "Unsymmetric/ck104.tar.gz")
# download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/young1c.tar.gz", "Unsymmetric/young1c.tar.gz")
# download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Bai/olm2000.tar.gz", "Unsymmetric/olm2000.tar.gz")
# download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/arc130.tar.gz", "Unsymmetric/arc130.tar.gz")
# download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/fs_680_3.tar.gz", "Unsymmetric/fs_680_3.tar.gz")
# download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/jpwh_991.tar.gz", "Unsymmetric/jpwh_991.tar.gz")
# download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/HB/plskz362.tar.gz", "Unsymmetric/plskz362.tar.gz")
# download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/DRIVCAV/cavity10.tar.gz", "Unsymmetric/cavity10.tar.gz")
# download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/FIDAP/ex35.tar.gz", "Unsymmetric/ex35.tar.gz")
# download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/thermomech_dK.tar.gz", "Unsymmetric/thermomech_dK.tar.gz")
# download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Hamm/scircuit.tar.gz", "Unsymmetric/scircuit.tar.gz")
# download_and_extract_matrix("https://suitesparse-collection-website.herokuapp.com/MM/Mallya/lhr01.tar.gz", "Unsymmetric/lhr01.tar.gz")