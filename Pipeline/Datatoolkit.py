import pandas as pd
import numpy as np
import segyio
import matplotlib.pyplot as plt
import os
import random
import cv2

class DatatoolKit():
    """
    Toolkit for loading, analyzing, and visualizing SEGY seismic data.


    Attributes:
    DIR (Path): Directory containing the SEGY file.
    filename (str): Name of the SEGY file.
    filename_full (str): Full path to the SEGY file.
    """
    def __init__(self, DIR:str, filename:str) -> None:
        self.DIR = DIR
        self.filename = filename
        self.filename_full = str(self.DIR / self.filename)
        pass

    def LoadSGY(self)->segyio.SegyFile:
        """
        Load SEGY file using segyio.


        Returns:
        segyio.SegyFile: Opened SEGY file object.
        """

        return segyio.open(self.filename_full,"r", ignore_geometry=True)
    
    def create_df(self, file: segyio.SegyFile, create_csv:bool = False) -> pd.DataFrame:
        """
        Create a pandas DataFrame containing trace metadata and amplitude arrays.


        Args:
        file (segyio.SegyFile): Loaded SEGY file.
        create_csv (bool): Whether to save the dataframe as CSV.


        Returns:
        pd.DataFrame: DataFrame with trace information.
        """
        rows= []
        
        #get all inlines and crosslines
        inlines   = file.attributes(segyio.TraceField.INLINE_3D)[:]
        crosslines = file.attributes(segyio.TraceField.CROSSLINE_3D)[:]

        #xreate the Dataframe
        for i in range(file.tracecount):
            amp = np.array(file.trace[i])
            inl = inlines[i]
            cross = crosslines[i]
            rows.append({
            "filename": self.filename,
            "trace": i,
            "inline": inl,
            "crossline": cross,
            "Amplitude": amp
        })
        df = pd.DataFrame(rows)

        #export to a csv if wished
        if create_csv:
            print("This could take a while because Datasets can be very large are you sure?")
            csv_opt = input("please type: 1 = yes , 2 = no")
            if csv_opt == 1:
                df.to_csv("inline_crossline_segyio.csv", index =False)
            else:
                return df
        return df
    
    def analyse_datetype(self, df:pd.DataFrame, create_pivot:bool = False) -> None:
        """
        Analyze whether dataset is 2D or 3D based on inline/crossline combinations.


        Args:
        df (pd.DataFrame): DataFrame created from the SEGY file.
        create_pivot (bool): Whether to print a pivot table of inline/crossline grid.
        """

        # unique inline and Croslline combinations
        combo_counts = df.groupby(["inline", "crossline"]).size().reset_index(name="count")

        # duplicated combinations
        duplicates = combo_counts[combo_counts["count"] > 1]

        # counting unique combinations
        unique_pairs = len(combo_counts)

        #counting number od traces
        total_traces = len(df)

        # number of inlines and crosslines
        inlines = df["inline"].nunique()
        crosslines = df["crossline"].nunique()

        # Analyse
        if len(duplicates) > 0:
            data_type = "2D"
        elif inlines == 1 or crosslines == 1:
            data_type = "2D"
        elif unique_pairs == total_traces:
            data_type = "3D (unique combinations)"
        else:
            data_type = "could not determine datatype"

        print(f"Inlines: {inlines}, Crosslines: {crosslines}")
        print(f"Traces: {total_traces}, Unique Pairs: {unique_pairs}")
        print(f"Number of duplicated combinations: {len(duplicates)}")
        print(f"Datatype: {data_type}")
        
        

        if create_pivot:
            pivot = df.pivot_table(index="inline", columns="crossline", aggfunc="size", fill_value=0)
            print(pivot)

            if pivot.shape[0] == inlines and pivot.shape[1] == crosslines:
                print("-------------------------------")
                print("Datatype: 3D")
    
        

    def plot_grid(self, df:pd.DataFrame) -> None:
        """Plot inline/crossline scatter grid.


        Args:
        df (pd.DataFrame): DataFrame containing inline and crossline info.
        """

        plt.figure(figsize=(8,6))
        plt.scatter(df["inline"], df["crossline"])
        plt.xlabel("Inline")
        plt.ylabel("Crossline")
        plt.title(f"Grid for file")
        plt.show()


    def plot_inline_cut_raw(self, df:pd.DataFrame, inlinenr:int) -> None:
        """
        Plot raw inline section.


        Args:
        df (pd.DataFrame): DataFrame of SEGY traces.
        inlinenr (int): Inline number to plot.
        """
        #Sort the crosslines with respect to the chosen Inline
        sub = df[df["inline"] == inlinenr].sort_values("crossline")
        traces = np.vstack(sub["Amplitude"].values)
        img = traces.T

        plt.figure(figsize=(10, 5))
        plt.imshow(img, cmap="grey", aspect="auto", origin="upper")
        plt.title(f"Inline {inlinenr} in {self.filename}")
        plt.xlabel("Crossline")
        plt.ylabel("Time sample")
        plt.colorbar(label="Amplitude")
        plt.show()

    def plot_crossline_cut_raw(self, df:pd.DataFrame, cross_nr:int) -> None:
        """
        Plot raw crossline section.


        Args:
        df (pd.DataFrame): DataFrame of SEGY traces.
        cross_nr (int): Crossline index to plot.
        """

        #Sort the inlines with respect to the chosen crossline
        sub = df[df["crossline"] == cross_nr].sort_values("inline")
        traces = np.vstack(sub["Amplitude"].values)
        img = traces.T

        plt.figure(figsize=(10, 5))
        plt.imshow(img, cmap="gray", aspect="auto", origin="upper")
        plt.title(f"Crossline {cross_nr} in {self.filename}")
        plt.xlabel("Inline")
        plt.ylabel("Time sample")
        plt.colorbar(label="Amplitude")
        plt.show()

    def plot_timeslice_cut_raw(self, df:pd.DataFrame, sample_index:int) -> None:
        """
        Plot a time slice for a given sample index.


        Args:
        df (pd.DataFrame): DataFrame of SEGY traces.
        sample_index (int): Sample index for extracting amplitudes.
        """
        
        # get Inlines and Crossline Indexes
        inlines = np.sort(df["inline"].unique())
        crosslines = np.sort(df["crossline"].unique())

        #create 2d Array filled with NAN
        mat = np.full((len(inlines), len(crosslines)), np.nan)

        #create lookup dictionary
        il_map = {v:i for i,v in enumerate(inlines)}
        cl_map = {v:i for i,v in enumerate(crosslines)}

        #put the Amplitudes to the right Position in 2D Array with respect to sample index
        for _, row in df.iterrows():
            i = il_map[row["inline"]]
            j = cl_map[row["crossline"]]
            mat[i, j] = row["Amplitude"][sample_index]

        plt.figure(figsize=(8,5))
        plt.imshow(np.nan_to_num(mat), cmap="gray", aspect="auto",
               extent=[crosslines.min(), crosslines.max(), inlines.max(), inlines.min()])
        plt.title(f"Time-Slice @ sample {sample_index} in {self.filename}")
        plt.xlabel("Crossline")
        plt.ylabel("Inline")
        plt.colorbar(label="Amplitude")
        plt.show()

    def create_images(self, file:segyio.SegyFile, df:pd.DataFrame, outdir:str, inline:bool = False, crossline:bool = False, timeslice:bool= False, resize:bool= False, Scale:tuple = None) -> None:
        """
        Export Radargeamms (inline, crossline, timeslice) as PNG images.


        Args:
        file (segyio.SegyFile): Loaded SEGY file.
        df (pd.DataFrame): DataFrame of trace data.
        outdir (str): Output directory for saving images.
        inline (bool): Save inline sections.
        crossline (bool): Save crossline sections.
        timeslice (bool): Save time slices.
        resize (bool): Whether to resize images.
        Scale (Optional[Tuple[int, int]]): (scale_h, scale_w) factors.
        """
    
        outdir = outdir
        filename = df["filename"].iloc[0]

        # Create Images for Inlines and save them
        if inline:
            inlines = sorted(df["inline"].unique())
            for inline_nr in inlines:
                sub = df[df["inline"] == inline_nr].sort_values("crossline")
                img = np.vstack(sub["Amplitude"].values).T
                out_path = os.path.join(outdir, f"{filename}_inline_{inline_nr}.png")
                if resize:
                    h, w = img.shape
                    scale_h , scale_w = Scale
                    img_resized = cv2.resize(img, (w* scale_w, h*scale_h), interpolation=cv2.INTER_LINEAR)
                    plt.imsave(out_path, img_resized, cmap="grey")
                else:
                    plt.imsave(out_path, img, cmap="grey")
        
            print(f"saved {inline_nr} images to {out_path}")

        #create Images for Crosslines and save them
        if crossline:
            crosslines = sorted(df["crossline"].unique())
            for crossline_nr in crosslines:
                sub = df[df["crossline"] == crossline_nr].sort_values("inline")
                img = np.vstack(sub["Amplitude"].values).T
                out_path = os.path.join(outdir, f"{filename}_crossline_{crossline_nr}.png")
                if resize:
                    h, w = img.shape
                    scale_h , scale_w = Scale
                    img_resized = cv2.resize(img, (w* scale_w, h*scale_h), interpolation=cv2.INTER_LINEAR)
                    plt.imsave(out_path, img_resized, cmap="grey")
                else:
                    plt.imsave(out_path, img, cmap="grey")
        
            print(f"saved {crossline_nr} images to {out_path}")
        
        #create Images for time slice (Top View) and save them
        if timeslice:

            bin_header = dict(file.bin)
            bin_header_dict = dict(bin_header)
            n_samples = int(bin_header_dict[segyio.BinField.Samples])

            inlines = np.sort(df["inline"].unique())
            crosslines = np.sort(df["crossline"].unique())
        

            il_map = {v:i for i,v in enumerate(inlines)}
            cl_map = {v:i for i,v in enumerate(crosslines)}

            for sample_index in range(n_samples):
                mat = np.full((len(inlines), len(crosslines)), np.nan)

                for _,row in df.iterrows():
                    i = il_map[row["inline"]]
                    j = cl_map[row["crossline"]]
                    mat[i, j] = row["Amplitude"][sample_index]
        
                img = np.nan_to_num(mat)
                out_path=os.path.join(outdir, f"{filename}_timeslice_{sample_index}.png")
                if resize:
                    h, w = img.shape
                    scale_h , scale_w = Scale
                    img_resized = cv2.resize(img, (w* scale_w, h*scale_h), interpolation=cv2.INTER_LINEAR)
                    plt.imsave(out_path, img_resized, cmap="grey")
                else:
                    plt.imsave(out_path, img, cmap="grey")
        
            print(f"saved {sample_index} images to {out_path}")
    
    
    def create_random_test_images(self, df:pd.DataFrame, file:segyio.SegyFile, outdir:str, test_inline_random:bool = False, test_crossline_random:bool = False, test_timeslice_random:bool = False,
                        number_random_inlines:int = 1, number_random_crosslines:int = 1, number_random_timeslices:int = 1
                        ):
        """Randomly export seismic test images (inline, crossline, timeslice).


        Args:
        df (pd.DataFrame): DataFrame of SEGY data.
        file (segyio.SegyFile): Loaded SEGY file.
        outdir (str): Output directory.
        test_inline_random (bool): Export random inline sections.
        test_crossline_random (bool): Export random crosslines.
        test_timeslice_random (bool): Export random time slices.
        number_random_inlines (int): Number of random inlines.
        number_random_crosslines (int): Number of random crosslines.
        number_random_timeslices (int): Number of random time slices.
        """
    
        outdir = outdir
        filename = df["filename"].iloc[0]

        bin_header = dict(file.bin)
        bin_header_dict = dict(bin_header)
        n_samples = int(bin_header_dict[segyio.BinField.Samples])

        inlines = sorted(df["inline"].unique())
        crosslines = sorted(df["crossline"].unique())

        il_map = {v: i for i, v in enumerate(inlines)}
        cl_map = {v: i for i, v in enumerate(crosslines)}

        # create number of random Inline images
        if test_inline_random:
            inlines = random.sample(inlines, min(number_random_inlines, len(inlines)))
            for inline_nr in inlines:
                sub = df[df["inline"] == inline_nr].sort_values("crossline")
                img = np.vstack(sub["Amplitude"].values).T
                out_path = os.path.join(outdir, f"{filename}_inline_{inline_nr}.png")
                plt.imsave(out_path, img, cmap="grey")
        
            print(f"saved {len(inlines)} images to {outdir}")
        
        #create number of random crossline images
        if test_crossline_random:
            crosslines = random.sample(crosslines, min(number_random_crosslines, len(crosslines)))
            for crossline_nr in crosslines:
                sub = df[df["crossline"] == crossline_nr].sort_values("inline")
                img = np.vstack(sub["Amplitude"].values).T
                out_path = os.path.join(outdir, f"{filename}_crossline_{crossline_nr}.png")
                plt.imsave(out_path, img, cmap="grey")
        
            print(f"saved {len(crosslines)} images to {outdir}")
        
        #create random number od timeslice images
        if test_timeslice_random:
            rand_timeslices = random.sample(range(n_samples), min(number_random_timeslices, n_samples))
            for sample_index in rand_timeslices:
                mat = np.full((len(inlines), len(crosslines)), np.nan)

                for _,row in df.iterrows():
                    i = il_map[row["inline"]]
                    j = cl_map[row["crossline"]]
                    mat[i, j] = row["Amplitude"][sample_index]
        
                img = np.nan_to_num(mat)
                out_path=os.path.join(outdir, f"{filename}_timeslice_{sample_index}.png")
                plt.imsave(out_path, img, cmap="grey")
        
            print(f"saved {len(rand_timeslices)} images to {outdir}")


        
        