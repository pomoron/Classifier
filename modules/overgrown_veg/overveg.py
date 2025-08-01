from . import det_vegpvmt, det_whiteline, det_overgrown
from datetime import datetime
import config.paths as paths

# detect vegetation and pavement
def test_run_overveg():
    start_time = datetime.now()
    det_vegpvmt.main()          # output: paths.detveg_fn
    end_vegpvmt = datetime.now()
    print(f"Vegetation and pavement detection done. \n Duration: {end_vegpvmt - start_time}")
    # from the output file, detect white lines on the pavement
    det_whiteline.main()        # output: paths.detveg_line_fn
    end_white = datetime.now()
    print(f"White line detection done. \n Duration: {end_white - end_vegpvmt}")
    # from the white line, detect depth and overgrown vegetation
    det_overgrown.main()        # output: paths.detveg_final_fn
    end_overgrown = datetime.now()
    print(f"Overgrown vegetation detection done. \n Duration: {end_overgrown - end_white}")

def run_overveg():
    
    # detect vegetation and pavement
    det_vegpvmt.main(
        input_dir=paths.detveg_indir,
        detveg_fn=paths.detveg_fn,
        want_plot=paths.detveg_want_plot,)          # output: paths.detveg_fn
    
    # from the output file, detect white lines on the pavement
    det_whiteline.main(
        input_dir=paths.detveg_indir,
        seg_list=paths.detveg_fn,
        output_fn=paths.detveg_line_fn,
    )        # output: paths.detveg_line_fn
    
    # from the white line, detect depth and overgrown vegetation
    det_overgrown.main(
        input_dir = paths.detveg_indir,
        seg_list = paths.detveg_fn,
        line_list = paths.detveg_line_fn,
        output_fn = paths.detveg_final_fn,
        output_vis = paths.detveg_overgrown_vis,
    )        

if __name__ == '__main__':
    run_overveg()