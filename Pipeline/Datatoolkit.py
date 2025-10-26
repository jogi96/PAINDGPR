import pandas as pd
import numpy as np
import segyio
import matplotlib.pyplot as plt


class DatatoolKit():
    def __init__(self, DIR:str, filename:str):
        self.DIR = DIR
        self.filename = filename
        self.filename_full = str(self.DIR / self.filename)
        pass

    def LoadSGY(self):
        return segyio.open(self.filename_full,"r", ignore_geometry=True)
    
    def create_df(self, f, create_csv =False):
        rows= []

        inlines   = f.attributes(segyio.TraceField.INLINE_3D)[:]
        crosslines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]

        for i in range(f.tracecount):
            amp = np.array(f.trace[i])
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
        if create_csv:
            print("This could take a while because Datasets can be very large are you sure?")
            csv_opt = input("please type: 1 = yes , 2 = no")
            if csv_opt == 1:
                df.to_csv("inline_crossline_segyio.csv", index =False)
            else:
                return df
        return df

    def plot_inline_raw(self, df, inlinenr):
    
    
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

    def plot_crossline_raw(self, df, cross_nr):

        sub = df[df["crossline"] == cross_nr].sort_values("inline")
        traces = np.vstack(sub["Amplitude"].values)
        img = traces.T

        plt.figure(figsize=(10, 5))
        plt.imshow(img, cmap="gray", aspect="auto", origin="upper")
        plt.title(f"Crossline {cross_nr}in {self.filename}")
        plt.xlabel("Inline")
        plt.ylabel("Time sample")
        plt.colorbar(label="Amplitude")
        plt.show()

    def plot_timeslice(self, df, sample_index):
        inlines = np.sort(df["inline"].unique())
        crosslines = np.sort(df["crossline"].unique())
        mat = np.full((len(inlines), len(crosslines)), np.nan)

        il_map = {v:i for i,v in enumerate(inlines)}
        cl_map = {v:i for i,v in enumerate(crosslines)}

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



        
        