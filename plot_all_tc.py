import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 雲端無介面繪圖必備
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from scipy.interpolate import PchipInterpolator
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from matplotlib.lines import Line2D
import tcmarkers

# ==================== 全局設定 ====================
SITE_BASE_URL = "https://zax41006-bot.github.io/TC-Track"

plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

INTENSITY_CONFIG = {
    "TD":   {"cn": "熱帶低氣壓", "color": "#FAD02C", "range": "(41‑62 kph)"},
    "TS":   {"cn": "熱帶風暴",   "color": "#0072C6", "range": "(63‑87 kph)"},
    "STS":  {"cn": "強烈熱帶風暴", "color": "#00A651", "range": "(88‑117 kph)"},
    "TY":   {"cn": "颱風",       "color": "#FF9900", "range": "(118‑149 kph)"},
    "STY":  {"cn": "強颱風",     "color": "#E60012", "range": "(150~184 kph)"},
    "SUTY": {"cn": "超強颱風",   "color": "#7B1FA2", "range": "(≥185 kph)"},
    "LPA":  {"cn": "低壓區",     "color": "#A0A0A0", "range": "(<41 kph)"},
    "MD":   {"cn": "低壓區",     "color": "#A0A0A0", "range": "(<41 kph)"},
    "EX":   {"cn": "溫帶氣旋",   "color": "#8D8D8D", "range": "(-)"}
}

def get_intensity_color(intensity_str):
    if intensity_str in INTENSITY_CONFIG:
        return INTENSITY_CONFIG[intensity_str]["color"]
    for k, v in INTENSITY_CONFIG.items():
        if v["cn"] == intensity_str:
            return v["color"]
    return "#A0A0A0"

def calc_distance(lng1, lat1, lng2, lat2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    raw_dist = R * c
    return int(round(raw_dist, -1))

def draw_chart(tc_id, tc_name):
    past_url = f"{SITE_BASE_URL}/past_track_{tc_id}.csv"
    forecast_csv = f"forecast_track_{tc_id}.csv"
    
    try:
        df_past = pd.read_csv(past_url)
        past_data = df_past[['datetime', 'lng', 'lat', 'wind', 'minimum central pressure','intensity']].values.tolist()
        curr = past_data[-1]
        base_time = pd.to_datetime(curr[0], dayfirst=True)
        curr_intensity_code = curr[5]
        
        if not os.path.exists(forecast_csv):
            print(f"⚠️ 找不到 {forecast_csv}，跳過 {tc_name}")
            return
            
        df_forecast = pd.read_csv(forecast_csv)
        forecast_data = []
        for _, row in df_forecast.iterrows():
            h = int(str(row['f_time']).replace('hr', ''))
            forecast_data.append([f"{h}hr", row['lng'], row['lat'], row['wind'], h, row['minimum central pressure'], row['intensity']])

        fig, ax = plt.subplots(figsize=(12, 11), subplot_kw={'projection': ccrs.PlateCarree()})
        
        lon_min, lon_max = 100.0, 180.0
        lat_min, lat_max = 0.0, 60.0
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor="#D0E8F5", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#FAF0E6", edgecolor="none", zorder=1)
        
        # 中國行政邊界
        china_land = NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='10m', facecolor='none')
        ax.add_feature(china_land, facecolor='none', edgecolor='#A0A0A0', linewidth=0.5, zorder=2)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='#888888', alpha=0.4, linestyle='--', zorder=1)
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = {'size': 9, 'weight':'bold'}
        gl.ylabel_style = {'size': 9, 'weight':'bold'}
        gl.xlocator = mticker.MultipleLocator(5)
        gl.ylocator = mticker.MultipleLocator(5)
        
        macau_lng, macau_lat = 113.55, 22.17
        ax.plot(macau_lng, macau_lat, marker='*', color='red', markersize=10, transform=ccrs.PlateCarree(), zorder=12, markeredgecolor='black', markeredgewidth=0.5)
        
        # ================= 澳門距離警戒圈 =================
        lon_span = lon_max - lon_min
        lat_span = lat_max - lat_min
        ranges_km = [800, 400] if (lon_span > 40 or lat_span > 40) else [800, 600, 400, 200, 100]
        theta = np.linspace(0, 2 * np.pi, 360)
        for r_km in ranges_km:
            r_lat_deg = r_km / 111.0
            r_lng_deg = r_km / (111.0 * np.cos(np.radians(macau_lat)))
            
            ring_min_lat = macau_lat - r_lat_deg
            ring_max_lat = macau_lat + r_lat_deg
            ring_min_lng = macau_lng - r_lng_deg
            ring_max_lng = macau_lng + r_lng_deg
            
            if ring_min_lat >= lat_min and ring_max_lat <= lat_max and ring_min_lng >= lon_min and ring_max_lng <= lon_max:
                d_lat = r_lat_deg * np.sin(theta)
                d_lng = r_lng_deg * np.cos(theta)
                
                ring_poly = Polygon(np.column_stack((macau_lng + d_lng, macau_lat + d_lat)))
                ax.add_geometries([ring_poly], ccrs.PlateCarree(), facecolor='none', edgecolor='#78909C', linewidth=0.6, linestyle='--', alpha=0.7, zorder=2)
                ax.text(macau_lng, macau_lat - r_lat_deg, f"{r_km}KM", transform=ccrs.PlateCarree(),
                        fontsize=8.0, color='#555555', fontweight='bold', ha='center', va='top', zorder=6)

        # ================= 過去路徑 =================
        past_lngs = [d[1] for d in past_data]
        past_lats = [d[2] for d in past_data]
        ax.plot(past_lngs, past_lats, color="black", lw=1.6, zorder=4)
        
        for d in past_data[:-1]:
            p_code = d[5]
            p_color = get_intensity_color(p_code)
            ax.plot(d[1], d[2], marker=tcmarkers.HU, ms=4.5, color=p_color, mec='k', mew=0.2, zorder=5)
            
        # ================= 預報路徑與誤差圈 =================
        if forecast_data:
            f_hs = [d[4] for d in forecast_data]
            all_h = [0] + f_hs
            all_ln = [curr[1]] + [d[1] for d in forecast_data]
            all_lt = [curr[2]] + [d[2] for d in forecast_data]
            all_er = [0] + [((h // 24) * 100 + (h % 24) * (100 / 24)) * (1 / 111) for h in f_hs]
            
            ih = np.linspace(0, max(all_h), 100)
            xi = PchipInterpolator(all_h, all_ln)(ih)
            yi = PchipInterpolator(all_h, all_lt)(ih)
            ri = PchipInterpolator(all_h, all_er)(ih)
            
            ps = [Polygon(np.dstack((xi[i] + ri[i] * np.cos(np.linspace(0, 2 * np.pi, 360)), yi[i] + ri[i] * np.sin(np.linspace(0, 2 * np.pi, 360))))[0]) for i in range(len(ih))]
            ax.add_geometries([unary_union([MultiPolygon([ps[i], ps[i + 1]]).convex_hull for i in range(len(ps) - 1)])], 
                              ccrs.PlateCarree(), fc="#d1d1d1", alpha=0.45, ec="#707070", lw=0.7, zorder=3)
            ax.plot(xi, yi, color="#002a7d", lw=2.0, ls='--', zorder=4)
            
        # ================= 預報節點與文字 =================
        target_hours = [24, 48, 72, 96, 120]
        for d in forecast_data:
            if d[4] not in target_hours: 
                continue
            col = get_intensity_color(d[6])
            ax.plot(d[1], d[2], marker=tcmarkers.HU, ms=6.0, color=col, mec='k', mew=0.4, zorder=10)
            
        ax.plot(curr[1], curr[2], marker=tcmarkers.HU, ms=6.5, color=get_intensity_color(curr_intensity_code), mec='k', mew=0.5, zorder=11)
        curr_dist = calc_distance(curr[1], curr[2], macau_lng, macau_lat)
        
        info_lines = [
            "現時資訊：",
            f"距離澳門：{curr_dist}公里",
            f"{base_time.strftime('%d/%m/%Y %H:%M')}",
            f"{curr_intensity_code}, {curr[3]}kph, {curr[4]}hPa",
            "──────────────────",
            "預報資訊："
        ]
        
        for d in forecast_data:
            if d[4] not in target_hours: 
                continue
            f_time = base_time + pd.Timedelta(hours=d[4])
            info_lines.append(f"{f_time.strftime('%d/%m/%Y %H:%M')}  {d[6]}")
            
        ax.text(0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes, ha='left', va='top',
                fontsize=10.0, fontweight='bold', color='#111111', 
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#B0BEC5", alpha=0.92, lw=0.8), zorder=25)
                
        max_forecast_hr = max([d[4] for d in forecast_data]) if forecast_data else 0
        fig.text(0.5, 0.950, f"熱帶氣旋「{tc_name}」路徑預報圖", ha='center', va='top', 
                 fontsize=20, fontweight='bold', color='#111111')
        
        fig.text(0.5, 0.910, f"預報時效：{max_forecast_hr}小時", ha='center', va='top', 
                 fontsize=13, fontweight='bold', color='#222222')
        
        ax.text(0.98, 0.98, "澳門氣象中心MMC 發佈", transform=ax.transAxes, ha='right', va='top', fontsize=14, fontweight='bold',
                color='#003399', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="none"), zorder=20)
                
        # ================= 圖例 =================
        leg_core = [
            Line2D([0], [0], color="black", lw=2.3, label='過去路徑'),
            Line2D([0], [0], color="#003399", lw=2.5, ls='--', label='預報路徑'),
            plt.Rectangle((0, 0), 1, 1, fc="#C8C8C8", alpha=0.45, ec="#707070", lw=1.2, label='預報誤差範圍'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markeredgecolor='k', markersize=14, label='澳門位置', ls='')
        ]
        leg_int = [
            Line2D([0], [0], marker=tcmarkers.HU, c=INTENSITY_CONFIG["LPA"]["color"], label='低壓區', ms=9.0, mec='k', mew=0.7, ls=''),
            Line2D([0], [0], marker=tcmarkers.HU, c=INTENSITY_CONFIG["TD"]["color"], label='熱帶低氣壓', ms=9.0, mec='k', mew=0.7, ls=''),
            Line2D([0], [0], marker=tcmarkers.HU, c=INTENSITY_CONFIG["TS"]["color"], label='熱帶風暴', ms=9.0, mec='k', mew=0.7, ls=''),
            Line2D([0], [0], marker=tcmarkers.HU, c=INTENSITY_CONFIG["STS"]["color"], label='強烈熱帶風暴', ms=9.0, mec='k', mew=0.7, ls=''),
            Line2D([0], [0], marker=tcmarkers.HU, c=INTENSITY_CONFIG["TY"]["color"], label='颱風', ms=9.0, mec='k', mew=0.7, ls=''),
            Line2D([0], [0], marker=tcmarkers.HU, c=INTENSITY_CONFIG["STY"]["color"], label='強颱風', ms=9.0, mec='k', mew=0.7, ls=''),
            Line2D([0], [0], marker=tcmarkers.HU, c=INTENSITY_CONFIG["SUTY"]["color"], label='超強颱風', ms=9.0, mec='k', mew=0.7, ls=''),
        ]
        
        # 🌟 動態偵測是否包含溫帶氣旋 (EX)
        has_ex = any(d[5] == "EX" for d in past_data) or any(d[6] == "EX" for d in forecast_data)
        if has_ex:
            leg_int.append(
                Line2D([0], [0], marker=tcmarkers.HU, c=INTENSITY_CONFIG["EX"]["color"], label='溫帶氣旋', ms=9.0, mec='k', mew=0.7, ls='')
            )
            
        leg_params = dict(loc='lower center', frameon=True, edgecolor='#CFD8DC', facecolor='white', framealpha=0.95)
        fig.legend(handles=leg_core, ncol=4, bbox_to_anchor=(0.5, 0.15), fontsize=12.0, prop={'weight':'bold'}, **leg_params)
        # 🌟 動態調整圖例欄數，確保加入 EX 後依然能完美排版
        fig.legend(handles=leg_int, ncol=len(leg_int), bbox_to_anchor=(0.5, 0.10), fontsize=12.0, prop={'weight':'bold'}, **leg_params)
        
        plt.subplots_adjust(bottom=0.24, top=0.86)
        os.makedirs("output", exist_ok=True)
        
        out_path = f"output/forecast_{tc_id}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig) 
        print(f"✅ {tc_name} ({tc_id}) 預報圖已生成至 {out_path}")
        
    except Exception as e:
        print(f"❌ {tc_name} ({tc_id}) 繪圖失敗: {e}")

if __name__ == "__main__":
    with open('tc_config.json', 'r', encoding='utf-8-sig') as f:
        tc_list = json.load(f)
    
    if len(sys.argv) > 1:
        target_id = sys.argv[1].upper()
        found = False
        for tc in tc_list:
            if tc['id'] == target_id:
                print(f"🎯 指定繪製單一氣旋: {tc['name']} ({tc['id']})")
                draw_chart(tc['id'], tc['name'])
                found = True
                break
        if not found:
            print(f"⚠️ 在 tc_config.json 中找不到 ID 為 {target_id} 的氣旋。")
    else:
        print(f"🌐 準備繪製所有氣旋，共 {len(tc_list)} 個...")
        for tc in tc_list:
            draw_chart(tc['id'], tc['name'])