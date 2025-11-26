from ultralytics import YOLO
import pandas as pd
import numpy as np
import os
import re
import json
import segyio
from sklearn.linear_model import LinearRegression
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objects as go
from matplotlib.cm import get_cmap

class Predictor():
    """
    Class providing utilities for object detection post-processing, bounding box
    matching, hyperbola fitting, and interactive 3D visualization of Detections and fitting.
    """

    def __init__(self) -> None:
        pass

    def extract_boxes_object_detection(self, results)->dict:
        
        all_boxes = {}

        for result in results:
            img_name = os.path.basename(result.path)
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                all_boxes[img_name] = boxes# Bounding boxes in (x1, y1, x2, y2) format
            else:
                all_boxes[img_name] = np.empty((0,4))       
        
        return all_boxes

    def match_detections(self,results, dist_trheshhold, save_path, export:bool = False)-> dict:
        global_boxes = []
        global_id = 0
        records = []

        def center_bbox(box):
            x1,y1,x2,y2 = box
            return ((x1+x2)/2, (y1+y2)/2)
    
        def euclidan_distance(c1, c2):
            return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
    
        def parse_filename(filename):
            base_name = os.path.basename(filename)
        
            #getting everything until SGY
            match_main = re.match(r"(.+?\.SGY)", base_name, re.IGNORECASE)
            filename_part = match_main.group(1) if match_main else base_name

            #getting cut_type and cut_number
            match_suffix = re.search(r"_(\w+)[_-](\d+)", base_name)
            cut_type = match_suffix.group(1) if match_suffix else "unknown"
            cut_number = int(match_suffix.group(2)) if match_suffix else -1

            return filename_part, cut_type, cut_number



        for result in results:
            img = os.path.basename(result.path)
            filename, cut_type, cut_number = parse_filename(img)

            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
           
            else:
                boxes = np.empty((0,4)) 
                  
        
            if hasattr(result, "masks") and result.masks is not None:
                polygons = [mask.tolist() for mask in result.masks.xy]
            
            else:
                polygons = []

            for i, box in enumerate(boxes):
                center = center_bbox(box)
                existing_id = None

                for gid, gcenter in global_boxes:
                    if euclidan_distance(center, gcenter)< dist_trheshhold:
                        existing_id = gid
                        break
            
                if existing_id is None:
                    existing_id = global_id
                    global_boxes.append((global_id,center))
                    global_id +=1

                polygon_data = json.dumps(polygons[i] if i < len(polygons) else [])
                bbox_data = json.dumps(box.tolist())
            

                records.append({
                    "filename": filename,
                    "cut_type": cut_type,
                    "cut_number": cut_number,
                    "bbox_number": existing_id,
                    "bbox_data": bbox_data,
                    "polygon_data": polygon_data
                })

        df = pd.DataFrame(records)
        if export:
            df.to_csv(f"{save_path}/results.csv")
       
        return df
    
    

    def validate_detections(self,matched_detections,results, csv:bool = False, csv_path:str = None):

        if csv:
            matched_df = pd.read_csv(csv_path, sep=",")
        else:
            matched_df = matched_detections

        number_detections_csv_total = matched_df.shape[0]
        number_of_detections_results = 0

        for result in results:
            if result.masks.xy is not None:
                number_of_detections_results += len(result.masks.xy)

        if number_detections_csv_total == number_of_detections_results:
            print("Detections YOLO matching with Matched Detections DF")
    
        else:
            print(f"Yolo Detections:{number_of_detections_results} Number of Matched Detections {number_detections_csv_total}")


    def fit_hyperbolas_on_every_cut(self,df, csv:bool=False, csv_path:str = None):
        if csv:
            df = pd.read_csv(csv_path)
        else:
            df = df

        fits = []
        for i, row in df.iterrows():
            #Fit Hyperbolas with multiplie Regression
            polygon =  json.loads(row["polygon_data"])
            bbox = row["bbox_number"]
            polygon = np.array(polygon)
            x = polygon[:,0]
            t = polygon[:,1]
            t = t

            y = t**2
            X = np.column_stack([np.ones_like(x), x, x**2])
            model = LinearRegression(fit_intercept=False)
            model.fit(X,y)

            print(f"fit per cut(polygon:{i} in bbox:{bbox} , coefficients{model.coef_})")
            b0, b1, b2 = model.coef_

            #Convert Coefficients to Hyperbel context
            v = np.sqrt((4/b2))
            x0 = (-b1) / (2*b2)
            t0 = np.sqrt(b0 -b1**2 / ( 4*b2))

            fits.append({
                "filename": row["filename"],
                "cut_number": row["cut_number"],
                "bbox_number":row["bbox_number"],
                "b0_Model": b0,
                "b1_Model": b1,
                "b2_Model": b2,
                "v":v,
                "x0":x0,
                "t0":t0,
                "X_Data":x,
                "t_data":t
            })
    
        fits_df = pd.DataFrame(fits)
    
        return fits_df
    

    def fit_hyperbolas_idealized(self,df, csv:bool=False, csv_path:str = None,):
        if csv:
            df = pd.read_csv(csv_path)
        else:
            df = df

        fits = []

        grouped = df.groupby("bbox_number")

        for bbox, group in grouped:
            #Fit Hyperbolas with multiplie Regression
            every_x = []
            every_t = []

            filenames = group["filename"].unique().tolist()
            cuts = group["cut_number"].unique().tolist()

            for i, row in group.iterrows():

                polygon =  json.loads(row["polygon_data"])
                polygon = np.array(polygon)
                x = polygon[:,0]
                t = polygon[:,1]

                every_x.append(x)
                every_t.append(t)
        
            every_x = np.hstack(every_x)
            every_t = np.hstack(every_t)

            y = every_t**2
            X = np.column_stack([np.ones_like(every_x), every_x, every_x**2])
            model = LinearRegression(fit_intercept=False)
            model.fit(X,y)

            print(f"idealized fit (fit for bbox: {bbox}, coefficients{model.coef_})")
            b0, b1, b2 = model.coef_

            #Convert Coefficients to Hyperbel context
            v = np.sqrt((4/b2))
            x0 = (-b1) / (2*b2)
            t0 = np.sqrt(b0 -b1**2 / ( 4*b2))

            fits.append({
                "bbox": bbox,
                "filename": filenames,
                "cut_number": cuts,
                "b0_Model": b0,
                "b1_Model": b1,
                "b2_Model": b2,
                "v":v,
                "x0":x0,
                "t0":t0,
                "X_Data":every_x,
                "t_data":every_t
            })
    
        fits_df = pd.DataFrame(fits)

    
        return fits_df
    
    def fit_hyperbolas(self,matched_detections, export_csv:bool = False, save_path:str = None):
    
        idelaized_fit = self.fit_hyperbolas_idealized(df =matched_detections)
        fits_per_cut = self.fit_hyperbolas_on_every_cut(df=matched_detections)

        if export_csv:
            idelaized_fit.to_csv(f"{save_path}/idealized_fit.csv", index=False)
            fits_per_cut.to_csv(f"{save_path}/fits_per_cut.csv", index=False)
            return idelaized_fit, fits_per_cut
        else:
            return idelaized_fit,fits_per_cut
        

    def get_axis_and_sample_rate(self,sgy_file,df):
        number_of_crosslines = len(df["crossline"].unique())
        number_of_inlines = len(df["inline"].unique())
        bin_header_file = dict(sgy_file.bin)
        sampels_per_trace = bin_header_file[segyio.BinField.Samples]
        sample_rate = bin_header_file[segyio.BinField.Interval]

        return number_of_crosslines, number_of_inlines, sampels_per_trace, sample_rate


    def help_function_physical_units(self,sample_rate,sample_rate_factor, number_of_crosslines, number_of_inlines, width_crosslines:int = None, width_inlines:int = None):
        #getting the time per pixel
        time_per_pixel = sample_rate / sample_rate_factor

        #getting distance per pixel for crossline
        distance_per_pixel_crosslines = width_crosslines/number_of_crosslines

        #getting distance per pixel for inlines
        distance_per_pixel_inlines = width_inlines/number_of_inlines

        return time_per_pixel, distance_per_pixel_crosslines, distance_per_pixel_inlines

    def convert_phys_params(self,df, dcrosslines, dtime, export_csv:bool = False, name_of_df:str=None, save_path:str = None):
        dcrosslines = float(dcrosslines)
        dtime = float(dtime)
        df = df.copy()
        df["x0_m"] = df["x0"] * dcrosslines
        df["t0_ns"] = df["t0"] * dtime
        df["v_m/ns"] = df["v"] * (dcrosslines/dtime)
        
        if export_csv:
            df.to_csv(f"{save_path}/{name_of_df}_phys.csv")
            return df
        else:
            return df
    
    
    def plot_hyperbolas_3d_interactive(self,fits_per_cut_df: pd.DataFrame,
                                       fit_idealized_df: pd.DataFrame,
                                       matched_detections: pd.DataFrame,
                                       physics_enabled: bool,
                                       sgy_file: str,
                                       df_from_DatatoolKit: pd.DataFrame,
                                       width_crosslines: float | None = None,
                                       width_inlines: float | None = None,
                                       sample_rate_factor: float | None = None
                                       ) -> None:
        """
        Plot 3D hyperbolas, apex points, fitted ideal surfaces and polygon detections.
        Supports pixel mode and physical unit mode (m, ns).

        Parameters
        ----------
        fits_per_cut_df : pd.DataFrame
            Hyperbola fit results per cut (raw + converted if physics_enabled=True).
        fit_idealized_df : pd.DataFrame
            Idealized hyperbola surfaces.
        matched_detections : pd.DataFrame
            Polygon detections with bbox and cut information.
        physics_enabled : bool
            If True → convert X→meters, Y→meters, Z→nanoseconds.
        sgy_file : str
            Input SGY file.
        df_from_DatatoolKit : pd.DataFrame
            Metadata dataframe.
        width_crosslines : float | None
            Width of entire crossline direction (meters).
        width_inlines : float | None
            Width of entire inline direction (meters).
        sample_rate_factor : float | None
            Optional multiplier for sample spacing.
        """

        # ============================================================
        # GET BASIC AXIS / SAMPLE INFORMATION
        # ============================================================
        number_of_crosslines,number_of_inlines,samples_per_trace,sample_rate= self.get_axis_and_sample_rate(sgy_file=sgy_file, 
                                                                                                            df=df_from_DatatoolKit)

        # ============================================================
        # PHYSICAL UNIT CONVERSION (IF ENABLED)
        # ============================================================
        if physics_enabled:
            if width_inlines is None or width_crosslines is None:
                raise ValueError("Width Inline and Width Crosslines must be set in physics mode.")

            # returns nanoseconds per sample, meters per pixel, meters per pixel
            time_per_pixel, dx, dy = self.help_function_physical_units(
                sample_rate=sample_rate,
                sample_rate_factor=sample_rate_factor,
                number_of_crosslines=number_of_crosslines,
                number_of_inlines=number_of_inlines,
                width_crosslines=width_crosslines,
                width_inlines=width_inlines)

            # convert fitted parameters
            fits_per_cut_df = self.convert_phys_params(fits_per_cut_df, dcrosslines=dx, dtime=time_per_pixel)
            fit_idealized_df = self.convert_phys_params(fit_idealized_df, dcrosslines=dx, dtime=time_per_pixel)

        # ============================================================
        # FIGURE SETUP
        # ============================================================
        fig = go.Figure()
        cmap = get_cmap("tab20")

        traces_hyper = []
        traces_apex = []
        traces_poly = []
        traces_surface = []
        traces_surface_apex = []

        # ============================================================
        # IDEALIZED SURFACES + APEX-LINES
        # ============================================================
        for _, row in fit_idealized_df.iterrows():

            bbox = int(row["bbox"])
            cuts = row["cut_number"]
            v_raw, x0_raw, t0_raw = row["v"], row["x0"], row["t0"]

            r, g, b, a = cmap(bbox % 20)
            col = f"rgba({255*r:.0f},{255*g:.0f},{255*b:.0f},0.7)"

            xmin, xmax = min(row["X_Data"]), max(row["X_Data"])
            cmin, cmax = min(cuts), max(cuts)

            # grid in pixel units
            xf = np.linspace(xmin, xmax, 60)
            yf = np.linspace(cmin, cmax, 60)
            Xg, Yg = np.meshgrid(xf, yf)
            Zg = np.sqrt((4 / v_raw**2) * (Xg - x0_raw)**2 + t0_raw**2)

            # convert
            if physics_enabled:
                xf = xf * dx
                yf = yf * dy
                Xg, Yg = np.meshgrid(xf, yf)
                Zg = Zg * time_per_pixel

            traces_surface.append(len(fig.data))
            fig.add_trace(go.Surface(
                x=Xg, y=Yg, z=Zg,
                opacity=0.5,
                showscale=False,
                colorscale=[[0, col], [1, col]],
                visible=False,
                showlegend=True
            ))

            # --------------------------------------
            # Apex line
            # --------------------------------------
            y_line = np.linspace(cmin, cmax, 50)
            x_line = np.full_like(y_line, x0_raw)
            z_line = np.full_like(y_line, t0_raw)

            if physics_enabled:
                x_line = x_line * dx
                y_line = y_line * dy
                z_line = z_line * time_per_pixel
                base = [x0_raw, row["x0_m"], t0_raw, row["t0_ns"], v_raw, row["v_m/ns"]]
                hovertemplate = (
                    "<b>Idealisiert Apex</b><br>"
                    "x0 = %{customdata[0]:.1f} px (%{customdata[1]:.3f} m)<br>"
                    "t0 = %{customdata[2]:.1f} samp (%{customdata[3]:.1f} ns)<br>"
                    "v = %{customdata[4]:.3f} px/samp (%{customdata[5]:.6f} m/ns)<br>"
                    "<extra></extra>"
                )
            else:
                base = [x0_raw, t0_raw, v_raw]
                hovertemplate = (
                    "<b>Idealisiert Apex</b><br>"
                    "x0 = %{customdata[0]:.1f} px<br>"
                    "t0 = %{customdata[1]:.1f} samp<br>"
                    "v = %{customdata[2]:.2f} px/samp<br>"
                    "<extra></extra>"
                )

            customdata = np.tile([base], (len(x_line), 1))

            traces_surface_apex.append(len(fig.data))
            fig.add_trace(go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode="lines",
                line=dict(color=col, width=5),
                visible=False,
                hovertemplate=hovertemplate,
                customdata=customdata,
                showlegend=True
            ))

        # ============================================================
        # HYPERBOLAS PER CUT + APEX POINT
        # ============================================================
        for _, row in fits_per_cut_df.iterrows():

            bbox = int(row["bbox_number"])
            cut = int(row["cut_number"])
            X = np.array(row["X_Data"])

            v_raw, x0_raw, t0_raw = row["v"], row["x0"], row["t0"]

            r, g, b, a = cmap(bbox % 20)
            col = f"rgba({255*r:.0f},{255*g:.0f},{255*b:.0f},{a})"

            # Hyperbola line (pixel space)
            xf = np.linspace(X.min(), X.max(), 150)
            yf = np.full_like(xf, cut)
            tf = np.sqrt((4 / v_raw**2) * (xf - x0_raw)**2 + t0_raw**2)

            # convert if needed
            if physics_enabled:
                xf = xf * dx
                yf = yf * dy
                tf = tf * time_per_pixel

            traces_hyper.append(len(fig.data))
            fig.add_trace(go.Scatter3d(
                x=xf, y=yf, z=tf,
                mode="lines",
                line=dict(color=col, width=4),
                visible=False,
                showlegend=False
            ))

            # Apex point
            if physics_enabled:
                x_ap = row["x0_m"]
                y_ap = cut * dy
                z_ap = row["t0_ns"]
                base = [x0_raw, row["x0_m"], t0_raw, row["t0_ns"], v_raw, row["v_m/ns"]]
                hovertemplate = (
                    "<b>Apex</b><br>"
                    "x0 = %{customdata[0]:.1f} px (%{customdata[1]:.3f} m)<br>"
                    "t0 = %{customdata[2]:.1f} samp (%{customdata[3]:.1f} ns)<br>"
                    "v = %{customdata[4]:.3f} px/samp (%{customdata[5]:.6f} m/ns)<br>"
                    "<extra></extra>"
                )
            else:
                x_ap = x0_raw
                y_ap = cut
                z_ap = t0_raw
                base = [x0_raw, t0_raw, v_raw]
                hovertemplate = (
                    "<b>Apex</b><br>"
                    "x0 = %{customdata[0]:.1f} px<br>"
                    "t0 = %{customdata[1]:.1f} samp<br>"
                    "v = %{customdata[2]:.2f} px/samp<br>"
                    "<extra></extra>"
                )

            traces_apex.append(len(fig.data))
            fig.add_trace(go.Scatter3d(
                x=[x_ap], y=[y_ap], z=[z_ap],
                mode="markers",
                marker=dict(size=6, color=col),
                visible=False,
                hovertemplate=hovertemplate,
                customdata=[base],
                showlegend=False
            ))

        # ============================================================
        # POLYGON CONTOURS
        # ============================================================
        for _, row in matched_detections.iterrows():

            try:
                poly = json.loads(row["polygon_data"])
            except Exception:
                continue

            if len(poly) < 2:
                continue

            xs = [p[0] for p in poly]
            zs = [p[1] for p in poly]
            cut = row["cut_number"]

            if physics_enabled:
                xs = [x * dx for x in xs]
                ys = [cut * dy] * len(xs)
                zs = [z * time_per_pixel for z in zs]
            else:
                ys = [cut] * len(xs)

            bbox = int(row["bbox_number"])
            r, g, b, a = cmap(bbox % 20)
            col = f"rgba({255*r:.0f},{255*g:.0f},{255*b:.0f},{a})"

            traces_poly.append(len(fig.data))
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=col, width=3),
                visible=False,
                showlegend=False
            ))

        # ============================================================
        # BUTTON MASK
        # ============================================================
        def mask(lst):
            vis = [False] * len(fig.data)
            for i in lst:
                vis[i] = True
            return vis

        # ============================================================
        # LAYOUT
        # ============================================================
        if physics_enabled:
            x_range = [0, number_of_crosslines * dx]
            y_range = [0, number_of_inlines * dy]
            z_range = [samples_per_trace * time_per_pixel, 0]
            z_title = "Zeit (ns)"
            x_title = "Crosslines (m)"
            y_title = "Inlines (m)"
        else:
            x_range = [0, number_of_crosslines]
            y_range = [0, number_of_inlines]
            z_range = [samples_per_trace, 0]
            z_title = "Zeit (Samples)"
            x_title = "Crosslines"
            y_title = "Inlines"

        fig.update_layout(
            title="<b>Detektionen und Hyperbeln</b>",
            scene=dict(
                xaxis=dict(title=x_title, range=x_range),
                yaxis=dict(title=y_title, range=y_range),
                zaxis=dict(title=z_title, range=z_range),
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(label="Polygone", method="update", args=[{"visible": mask(traces_poly)}]),
                        dict(label="Hyperbeln Pro Cut", method="update", args=[{"visible": mask(traces_hyper)}]),
                        dict(label="Apex Pro Hyperbel", method="update", args=[{"visible": mask(traces_apex)}]),
                        dict(label="Idealisiert (Flächen)", method="update", args=[{"visible": mask(traces_surface)}]),
                        dict(label="Apex Idealisiert", method="update", args=[{"visible": mask(traces_surface_apex)}]),
                    ],
                    x=0,
                    y=0.85
                )
            ],
            height=700
        )

        fig.show()






    




    




    
    

    
    


        
       

    
    
    
    
   
    


        