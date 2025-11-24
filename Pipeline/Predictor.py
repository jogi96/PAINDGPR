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
    def __init__(self):
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

    def match_detections(self,results, dist_trheshhold, out_path, export:bool = False)-> dict:
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
            df.to_csv(f"{out_path}/results.csv")
       
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
            print("Detections YOLO matching with Matched Detections")
    
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

            print(f"polygon:{i} in bbox:{bbox} , coefficients{model.coef_}")
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

            print(f"fit for bbox: {bbox}, coefficients{model.coef_}")
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

    def convert_phys_params(self,df, dcrosslines, dtime):
        dcrosslines = float(dcrosslines)
        dtime = float(dtime)
        df = df.copy()
        df["x0_m"] = df["x0"] * dcrosslines
        df["t0_ns"] = df["t0"] * dtime
        df["v_m/ns"] = df["v"] * (dcrosslines/dtime)
        return df
    
    
    def plot_hyperbolas_3d_interactive(self,fits_per_cut_df, fit_idealized_df, matched_detections,physics_enabled,sgy_file, df_from_DatatoolKit,width_crosslines=None, width_inlines=None):

        number_of_crosslines, number_of_inlines, sampels_per_trace, sample_rate = self.get_axis_and_sample_rate(sgy_file=sgy_file, df=df_from_DatatoolKit)

        if physics_enabled:
            if width_inlines is None or width_crosslines is None:
                raise ValueError("Width Inline or Width crosslines cant be empty when physics enabled")

            time_per_pixel, distance_per_pixel_crosslines, distance_per_pixel_inlines = \
                self.help_function_physical_units(
                    sample_rate=sample_rate,
                    number_of_crosslines=number_of_crosslines,
                    number_of_inlines=number_of_inlines,
                    width_crosslines=width_crosslines,
                    width_inlines=width_inlines
                )

            fits_per_cut_df = self.convert_phys_params(
                df=fits_per_cut_df,
                dcrosslines=distance_per_pixel_crosslines,
                dtime=time_per_pixel
            )

        fit_idealized_df = self.convert_phys_params(
                df=fit_idealized_df,
                dcrosslines=distance_per_pixel_crosslines,
                dtime=time_per_pixel
            )

        # FIGURE
        fig = go.Figure()
        cmap = get_cmap("tab20")

        traces_hyper = []
        traces_apex = []
        traces_poly = []
        traces_surface = []
        traces_surface_apex = []

        # ----------------------------------------------------
        # IDEALISIERTE FLÄCHEN + APEX-LINIEN
        # ----------------------------------------------------
        for _, row in fit_idealized_df.iterrows():

            bbox = int(row["bbox"])
            cuts = row["cut_number"]
            v_raw, x0_raw, t0_raw = row["v"], row["x0"], row["t0"]

            r, g, b, a = cmap(bbox % 20)
            col = f"rgba({255*r:.0f},{255*g:.0f},{255*b:.0f},0.7)"

            xmin, xmax = min(row["X_Data"]), max(row["X_Data"])
            cmin, cmax = min(cuts), max(cuts)

            xf = np.linspace(xmin, xmax, 60)
            yf = np.linspace(cmin, cmax, 60)

            Xg, Yg = np.meshgrid(xf, yf)
            Zg = np.sqrt((4 / v_raw**2) * (Xg - x0_raw)**2 + t0_raw**2)

            traces_surface.append(len(fig.data))
            fig.add_trace(go.Surface(
                x=Xg, y=Yg, z=Zg,
                opacity=0.5,
                showscale=False,
                colorscale=[[0, col], [1, col]],
                visible=False
            ))

            # ===== APEX-LINIE =====
            y_line = np.linspace(cmin, cmax, 50)
            x_line = np.full_like(y_line, x0_raw)
            z_line = np.full_like(y_line, t0_raw)

            # Hover Daten
            if physics_enabled:
                x0_m = row["x0_m"]
                t0_ns = row["t0_ns"]
                v_m_ns = row["v_m/ns"]

                hovertemplate = (
                    "<b>Idealisiert Apex</b><br>"
                    "x0 = %{customdata[0]:.1f} px (%{customdata[1]:.3f} m)<br>"
                    "t0 = %{customdata[2]:.1f} samp (%{customdata[3]:.1f} ns)<br>"
                    "v = %{customdata[4]:.3f} px/samp (%{customdata[5]:.6f} m/ns)<br>"
                    "<extra></extra>"
                )

                base = [x0_raw, x0_m, t0_raw, t0_ns, v_raw, v_m_ns]

            else:
                hovertemplate = (
                    "<b>Idealisiert Apex</b><br>"
                    "x0 = %{customdata[0]:.1f} px<br>"
                    "t0 = %{customdata[1]:.1f} samp<br>"
                    "v = %{customdata[2]:.2f} px/samp<br>"
                    "<extra></extra>"
                )

                base = [x0_raw, t0_raw, v_raw]

            # customdata für jeden Punkt replizieren
            customdata = np.tile([base], (len(x_line), 1))

            traces_surface_apex.append(len(fig.data))
            fig.add_trace(
                go.Scatter3d(
                    x=x_line, y=y_line, z=z_line,
                    mode="lines",
                    line=dict(color=col, width=5),
                    visible=False,
                    hovertemplate=hovertemplate,
                    hoverinfo="text",
                    customdata=customdata
                )
            )

        # ----------------------------------------------------
        # HYPERBELN PRO CUT + APEX-PUNKT
        # ----------------------------------------------------
        for _, row in fits_per_cut_df.iterrows():

            bbox = int(row["bbox_number"])
            cut = int(row["cut_number"])
            X = np.array(row["X_Data"])

            v_raw, x0_raw, t0_raw = row["v"], row["x0"], row["t0"]

            r, g, b, a = cmap(bbox % 20)
            col = f"rgba({255*r:.0f},{255*g:.0f},{255*b:.0f},{a})"

            xf = np.linspace(X.min(), X.max(), 150)
            yf = np.full_like(xf, cut)
            tf = np.sqrt((4 / v_raw**2) * (xf - x0_raw)**2 + t0_raw**2)

            traces_hyper.append(len(fig.data))
            fig.add_trace(go.Scatter3d(
                x=xf, y=yf, z=tf,
                mode="lines",
                line=dict(color=col, width=4),
                visible=True
            ))

            # ===== APEX-PUNKT =====
            if physics_enabled:
                x0_m = row["x0_m"]
                t0_ns = row["t0_ns"]
                v_m_ns = row["v_m/ns"]

                hovertemplate = (
                    "<b>Apex</b><br>"
                    "x0 = %{customdata[0]:.1f} px (%{customdata[1]:.3f} m)<br>"
                    "t0 = %{customdata[2]:.1f} samp (%{customdata[3]:.1f} ns)<br>"
                    "v = %{customdata[4]:.3f} px/samp (%{customdata[5]:.6f} m/ns)<br>"
                    "<extra></extra>"
                )

                customdata = [[x0_raw, x0_m, t0_raw, t0_ns, v_raw, v_m_ns]]

            else:
                hovertemplate = (
                    "<b>Apex</b><br>"
                    "x0 = %{customdata[0]:.1f} px<br>"
                    "t0 = %{customdata[1]:.1f} samp<br>"
                    "v = %{customdata[2]:.2f} px/samp<br>"
                    "<extra></extra>"
                )

                customdata = [[x0_raw, t0_raw, v_raw]]

            traces_apex.append(len(fig.data))
            fig.add_trace(go.Scatter3d(
                x=[x0_raw], y=[cut], z=[t0_raw],
                mode="markers",
                marker=dict(size=6, color=col),
                visible=False,
                hovertemplate=hovertemplate,
                customdata=customdata
            ))

        # ----------------------------------------------------
        # POLYGON-KONTUREN
        # ----------------------------------------------------
        for _, row in matched_detections.iterrows():
            try:
                poly = json.loads(row["polygon_data"])
            except:
                continue

            xs = [p[0] for p in poly]
            zs = [p[1] for p in poly]
            ys = [row["cut_number"]] * len(xs)

            if len(xs) < 2:
                continue

            bbox = int(row["bbox_number"])
            r, g, b, a = cmap(bbox % 20)
            col = f"rgba({255*r:.0f},{255*g:.0f},{255*b:.0f},{a})"

            traces_poly.append(len(fig.data))
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=col, width=3),
                visible=False
            ))

        # ----------------------------------------------------
        # BUTTONMASKE
        # ----------------------------------------------------
        def mask(traces):
            vis = [False] * len(fig.data)
            for t in traces:
                vis[t] = True
            return vis

        # ----------------------------------------------------
        # LAYOUT
        # ----------------------------------------------------
        fig.update_layout(
            title="<b>Detektionen und Hyperbeln</b>",
            scene=dict(
                xaxis=dict(title="Crosslines"),
                yaxis=dict(title="Inlines"),
                zaxis=dict(title="Zeit (Samples)", range=[sampels_per_trace, 0])
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(label="Polygone Detektiert",
                            method="update",
                            args=[{"visible": mask(traces_poly)}]),
                        dict(label="Hyperbeln Pro Cut",
                            method="update",
                            args=[{"visible": mask(traces_hyper)}]),
                        dict(label="Scheitelpunkte Pro Hyperbel",
                            method="update",
                            args=[{"visible": mask(traces_apex)}]),
                        dict(label="Hyperbeln Idealisiert",
                            method="update",
                            args=[{"visible": mask(traces_surface)}]),
                        dict(label="Scheitelpunkte Idealisiert",
                            method="update",
                            args=[{"visible": mask(traces_surface_apex)}]),
                    ],
                    x=0, y=0.85
                )
            ],
            height=700
        )

        fig.show()





    




    




    
    

    
    


        
       

    
    
    
    
   
    


        