"""
Bias Analysis Using Protected Attributes
For AI Bias Mitigation Research Project

This script analyzes bias in resume fit scores using ACTUAL protected
attributes (nationality, age, gender) rather than inferred demographics.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# Country code to region mapping
COUNTRY_TO_REGION = {
    # Africa
    'NG': 'Africa', 'EG': 'Africa', 'ZA': 'Africa', 'KE': 'Africa',
    'SN': 'Africa', 'SL': 'Africa', 'MA': 'Africa', 'GH': 'Africa',
    # Middle East
    'AE': 'Middle East', 'SA': 'Africa', 'OM': 'Middle East',
    'QA': 'Middle East', 'KW': 'Middle East', 'BH': 'Middle East',
    # South Asia
    'IN': 'South Asia', 'PK': 'South Asia', 'BD': 'South Asia',
    'LK': 'South Asia', 'NP': 'South Asia',
    # East Asia
    'JP': 'East Asia', 'KR': 'East Asia', 'CN': 'East Asia',
    'TW': 'East Asia', 'HK': 'East Asia', 'SG': 'East Asia',
    # Europe
    'GB': 'Europe', 'DE': 'Europe', 'FR': 'Europe', 'ES': 'Europe',
    'IT': 'Europe', 'NL': 'Europe', 'PT': 'Europe', 'SE': 'Europe',
    'NO': 'Europe', 'DK': 'Europe', 'FI': 'Europe', 'PL': 'Europe',
    'RU': 'Europe', 'UA': 'Europe', 'CH': 'Europe', 'AT': 'Europe',
    'BE': 'Europe', 'IE': 'Europe', 'GR': 'Europe',
    # Americas
    'US': 'North America', 'CA': 'North America',
    'MX': 'Latin America', 'BR': 'Latin America', 'AR': 'Latin America',
    'CO': 'Latin America', 'CL': 'Latin America', 'PE': 'Latin America',
    # Oceania
    'AU': 'Oceania', 'NZ': 'Oceania',
}

COUNTRY_NAMES = {
    'NG': 'Nigeria', 'IN': 'India', 'AE': 'UAE', 'EG': 'Egypt',
    'JP': 'Japan', 'KR': 'South Korea', 'US': 'USA', 'GB': 'UK',
    'DE': 'Germany', 'FR': 'France', 'BR': 'Brazil', 'MX': 'Mexico',
    'ZA': 'South Africa', 'KE': 'Kenya', 'AU': 'Australia',
    'CA': 'Canada', 'ES': 'Spain', 'IT': 'Italy', 'RU': 'Russia',
    'CN': 'China', 'SG': 'Singapore', 'NL': 'Netherlands',
    'SE': 'Sweden', 'NO': 'Norway', 'PT': 'Portugal', 'PL': 'Poland',
    'OM': 'Oman', 'SN': 'Senegal', 'SL': 'Sierra Leone',
}


class ProtectedBiasAnalyzer:
    """
    Analyze bias using actual protected attributes.
    """

    def __init__(self, scores_df, protected_df):
        # Merge datasets on Candidate ID
        self.df = scores_df.merge(protected_df, on='Candidate ID', how='inner')
        print(f"Merged {len(self.df)} candidates with protected attributes")

        # Add derived columns
        self.df['region'] = self.df['nationality'].map(
            lambda x: COUNTRY_TO_REGION.get(x, 'Other')
        )
        self.df['country_name'] = self.df['nationality'].map(
            lambda x: COUNTRY_NAMES.get(x, x)
        )

        # Clean gender column
        self.df['gender'] = self.df['gender'].str.lower().str.strip()

    def overview(self):
        """Print dataset overview."""
        print("\n" + "=" * 60)
        print("DATASET OVERVIEW")
        print("=" * 60)
        print(f"Total candidates: {len(self.df)}")
        print(f"\nGender distribution:")
        print(self.df['gender'].value_counts())
        print(f"\nAge group distribution:")
        print(self.df['age_group'].value_counts().sort_index())
        print(f"\nRegion distribution:")
        print(self.df['region'].value_counts())
        print(f"\nTop 10 nationalities:")
        print(self.df['nationality'].value_counts().head(10))

    def analyze_by_attribute(self, attribute, score_col='fit_score'):
        """Analyze scores by a protected attribute."""
        stats_df = self.df.groupby(attribute)[score_col].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(4)
        stats_df = stats_df.sort_values('mean', ascending=False)
        return stats_df

    def statistical_tests(self, attribute, score_col='fit_score'):
        """Run statistical tests for group differences."""
        groups = self.df.groupby(attribute)[score_col].apply(list).to_dict()

        # Filter groups with sufficient samples
        groups = {k: v for k, v in groups.items() if len(v) >= 5}

        if len(groups) < 2:
            return {"error": "Not enough groups with sufficient samples"}

        results = {}

        # ANOVA (if more than 2 groups)
        if len(groups) > 2:
            f_stat, p_value = stats.f_oneway(*groups.values())
            results['anova'] = {
                'f_statistic': round(f_stat, 4),
                'p_value': round(p_value, 6),
                'significant_at_0.05': p_value < 0.05,
                'significant_at_0.01': p_value < 0.01
            }

        # Kruskal-Wallis (non-parametric alternative)
        if len(groups) > 2:
            h_stat, p_value = stats.kruskal(*groups.values())
            results['kruskal_wallis'] = {
                'h_statistic': round(h_stat, 4),
                'p_value': round(p_value, 6),
                'significant_at_0.05': p_value < 0.05
            }

        # Effect size (eta-squared for ANOVA)
        if 'anova' in results:
            grand_mean = self.df[score_col].mean()
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups.values())
            ss_total = sum((x - grand_mean)**2 for g in groups.values() for x in g)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            results['effect_size'] = {
                'eta_squared': round(eta_squared, 4),
                'interpretation': 'small' if eta_squared < 0.06 else 'medium' if eta_squared < 0.14 else 'large'
            }

        return results

    def fairness_metrics(self, attribute, score_col='fit_score', top_percentile=0.2):
        """Compute fairness metrics for a protected attribute."""
        threshold = self.df[score_col].quantile(1 - top_percentile)
        self.df['selected'] = self.df[score_col] >= threshold

        metrics = {}

        # Selection rates by group
        selection_rates = self.df.groupby(attribute)['selected'].mean()
        metrics['selection_rates'] = selection_rates.round(4).to_dict()

        # Demographic Parity
        max_rate = selection_rates.max()
        min_rate = selection_rates.min()
        metrics['demographic_parity'] = {
            'max_selection_rate': round(max_rate, 4),
            'min_selection_rate': round(min_rate, 4),
            'disparity_ratio': round(min_rate / max_rate, 4) if max_rate > 0 else 0,
            'four_fifths_rule': (min_rate / max_rate >= 0.8) if max_rate > 0 else True
        }

        # Score Parity
        avg_scores = self.df.groupby(attribute)[score_col].mean()
        metrics['score_parity'] = {
            'max_avg_score': round(avg_scores.max(), 4),
            'min_avg_score': round(avg_scores.min(), 4),
            'score_gap': round(avg_scores.max() - avg_scores.min(), 4),
            'relative_gap_pct': round((avg_scores.max() - avg_scores.min()) / avg_scores.max() * 100, 2)
        }

        return metrics

    def intersectional_analysis(self, attr1, attr2, score_col='fit_score'):
        """Analyze bias at the intersection of two attributes."""
        # Create combined attribute
        self.df['intersection'] = self.df[attr1].astype(str) + ' / ' + self.df[attr2].astype(str)

        # Get statistics
        stats_df = self.df.groupby('intersection')[score_col].agg([
            'count', 'mean', 'std'
        ]).round(4)
        stats_df = stats_df[stats_df['count'] >= 5]  # Filter small groups
        stats_df = stats_df.sort_values('mean', ascending=False)

        return stats_df

    def generate_visualizations(self, output_dir='.'):
        """Generate comprehensive bias visualizations."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        fig.suptitle('Bias Analysis: Fit Scores by Protected Attributes', fontsize=14, fontweight='bold')

        # 1. Score distribution by gender
        ax = axes[0, 0]
        gender_data = self.df[self.df['gender'].isin(['male', 'female'])]
        sns.boxplot(data=gender_data, x='gender', y='fit_score', ax=ax, palette='Set2')
        ax.set_title('Fit Scores by Gender')
        ax.set_xlabel('Gender')
        ax.set_ylabel('Fit Score')
        # Add means
        means = gender_data.groupby('gender')['fit_score'].mean()
        for i, gender in enumerate(['female', 'male']):
            if gender in means.index:
                ax.scatter(i, means[gender], color='red', s=100, zorder=5, marker='D')

        # 2. Score distribution by age group
        ax = axes[0, 1]
        age_order = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
        age_order = [a for a in age_order if a in self.df['age_group'].values]
        sns.boxplot(data=self.df, x='age_group', y='fit_score', order=age_order, ax=ax, palette='Blues')
        ax.set_title('Fit Scores by Age Group')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Fit Score')

        # 3. Score distribution by region
        ax = axes[0, 2]
        region_order = self.df.groupby('region')['fit_score'].mean().sort_values(ascending=False).index
        sns.boxplot(data=self.df, x='region', y='fit_score', order=region_order, ax=ax, palette='Set3')
        ax.set_title('Fit Scores by Region')
        ax.tick_params(axis='x', rotation=45)

        # 4. Mean scores by nationality (top 15)
        ax = axes[1, 0]
        nat_means = self.df.groupby('country_name')['fit_score'].agg(['mean', 'count'])
        nat_means = nat_means[nat_means['count'] >= 5].sort_values('mean', ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(nat_means)))
        nat_means['mean'].plot(kind='barh', ax=ax, color=colors)
        ax.axvline(x=self.df['fit_score'].mean(), color='red', linestyle='--', label='Overall Mean')
        ax.set_title('Average Fit Score by Nationality (n≥5)')
        ax.set_xlabel('Average Fit Score')
        ax.legend()

        # 5. Selection rate by gender (top 20%)
        ax = axes[1, 1]
        threshold = self.df['fit_score'].quantile(0.8)
        self.df['top_20'] = self.df['fit_score'] >= threshold
        selection_by_gender = self.df.groupby('gender')['top_20'].mean()
        colors = ['#ff6b6b' if rate < 0.16 else '#4ecdc4' for rate in selection_by_gender]
        selection_by_gender.plot(kind='bar', ax=ax, color=colors)
        ax.axhline(y=0.2, color='green', linestyle='--', label='Expected (20%)')
        ax.set_title('Selection Rate (Top 20%) by Gender')
        ax.set_ylabel('Selection Rate')
        ax.tick_params(axis='x', rotation=0)
        ax.legend()

        # 6. Selection rate by region
        ax = axes[1, 2]
        selection_by_region = self.df.groupby('region')['top_20'].mean().sort_values(ascending=True)
        colors = ['#ff6b6b' if rate < 0.16 else '#4ecdc4' for rate in selection_by_region]
        selection_by_region.plot(kind='barh', ax=ax, color=colors)
        ax.axvline(x=0.2, color='green', linestyle='--', label='Expected (20%)')
        ax.set_title('Selection Rate (Top 20%) by Region')
        ax.set_xlabel('Selection Rate')
        ax.legend()

        # 7. Heatmap: Gender x Region
        ax = axes[2, 0]
        pivot = self.df.pivot_table(values='fit_score', index='gender', columns='region', aggfunc='mean')
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, center=self.df['fit_score'].mean())
        ax.set_title('Mean Score: Gender × Region')

        # 8. Heatmap: Age x Gender
        ax = axes[2, 1]
        pivot = self.df.pivot_table(values='fit_score', index='age_group', columns='gender', aggfunc='mean')
        pivot = pivot.reindex(['20-29', '30-39', '40-49', '50-59', '60-69', '70-79'])
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, center=self.df['fit_score'].mean())
        ax.set_title('Mean Score: Age × Gender')

        # 9. Score distribution histogram with group overlay
        ax = axes[2, 2]
        for gender in ['male', 'female']:
            subset = self.df[self.df['gender'] == gender]['fit_score']
            ax.hist(subset, bins=25, alpha=0.5, label=f'{gender.title()} (n={len(subset)})', density=True)
        ax.axvline(x=self.df['fit_score'].mean(), color='black', linestyle='--', label='Overall Mean')
        ax.set_title('Score Distribution by Gender')
        ax.set_xlabel('Fit Score')
        ax.set_ylabel('Density')
        ax.legend()

        plt.tight_layout()
        output_path = os.path.join(output_dir, 'bias_analysis_protected.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualizations saved to '{output_path}'")

        return fig

    def generate_report(self):
        """Generate comprehensive bias report."""
        report = []
        report.append("=" * 70)
        report.append("BIAS ANALYSIS REPORT - PROTECTED ATTRIBUTES")
        report.append("Resume Fit Score Analysis for Bias Mitigation")
        report.append("=" * 70)

        # Overview
        report.append("\n## DATASET OVERVIEW")
        report.append(f"Total candidates: {len(self.df)}")
        report.append(f"Score range: {self.df['fit_score'].min():.3f} - {self.df['fit_score'].max():.3f}")
        report.append(f"Mean score: {self.df['fit_score'].mean():.3f} (std: {self.df['fit_score'].std():.3f})")

        concerns = []

        # ===== GENDER ANALYSIS =====
        report.append("\n" + "=" * 70)
        report.append("## GENDER ANALYSIS")
        report.append("=" * 70)

        gender_stats = self.analyze_by_attribute('gender')
        report.append("\nStatistics by gender:")
        report.append(gender_stats.to_string())

        gender_tests = self.statistical_tests('gender')
        if 'anova' in gender_tests:
            report.append(f"\nANOVA: F={gender_tests['anova']['f_statistic']}, p={gender_tests['anova']['p_value']}")
            if gender_tests['anova']['significant_at_0.05']:
                concerns.append("Gender: Statistically significant score differences detected")

        gender_fairness = self.fairness_metrics('gender')
        report.append(f"\nFairness Metrics:")
        report.append(f"  Selection rates: {gender_fairness['selection_rates']}")
        report.append(f"  Disparity ratio: {gender_fairness['demographic_parity']['disparity_ratio']}")
        report.append(f"  Four-fifths rule passed: {gender_fairness['demographic_parity']['four_fifths_rule']}")
        report.append(f"  Score gap: {gender_fairness['score_parity']['score_gap']} ({gender_fairness['score_parity']['relative_gap_pct']}%)")

        if not gender_fairness['demographic_parity']['four_fifths_rule']:
            concerns.append("Gender: Fails four-fifths rule for selection rates")

        # ===== AGE ANALYSIS =====
        report.append("\n" + "=" * 70)
        report.append("## AGE GROUP ANALYSIS")
        report.append("=" * 70)

        age_stats = self.analyze_by_attribute('age_group')
        report.append("\nStatistics by age group:")
        report.append(age_stats.to_string())

        age_tests = self.statistical_tests('age_group')
        if 'anova' in age_tests:
            report.append(f"\nANOVA: F={age_tests['anova']['f_statistic']}, p={age_tests['anova']['p_value']}")
            if age_tests['anova']['significant_at_0.05']:
                concerns.append("Age: Statistically significant score differences between age groups")
            if 'effect_size' in age_tests:
                report.append(f"Effect size (eta²): {age_tests['effect_size']['eta_squared']} ({age_tests['effect_size']['interpretation']})")

        age_fairness = self.fairness_metrics('age_group')
        report.append(f"\nFairness Metrics:")
        report.append(f"  Disparity ratio: {age_fairness['demographic_parity']['disparity_ratio']}")
        report.append(f"  Four-fifths rule passed: {age_fairness['demographic_parity']['four_fifths_rule']}")
        report.append(f"  Score gap: {age_fairness['score_parity']['score_gap']} ({age_fairness['score_parity']['relative_gap_pct']}%)")

        if not age_fairness['demographic_parity']['four_fifths_rule']:
            concerns.append("Age: Fails four-fifths rule - potential age discrimination")

        # ===== REGION ANALYSIS =====
        report.append("\n" + "=" * 70)
        report.append("## REGIONAL ANALYSIS")
        report.append("=" * 70)

        region_stats = self.analyze_by_attribute('region')
        report.append("\nStatistics by region:")
        report.append(region_stats.to_string())

        region_tests = self.statistical_tests('region')
        if 'anova' in region_tests:
            report.append(f"\nANOVA: F={region_tests['anova']['f_statistic']}, p={region_tests['anova']['p_value']}")
            if region_tests['anova']['significant_at_0.05']:
                concerns.append("Region: Statistically significant score differences between regions")
            if 'effect_size' in region_tests:
                report.append(f"Effect size (eta²): {region_tests['effect_size']['eta_squared']} ({region_tests['effect_size']['interpretation']})")

        region_fairness = self.fairness_metrics('region')
        report.append(f"\nFairness Metrics:")
        report.append(f"  Disparity ratio: {region_fairness['demographic_parity']['disparity_ratio']}")
        report.append(f"  Four-fifths rule passed: {region_fairness['demographic_parity']['four_fifths_rule']}")
        report.append(f"  Score gap: {region_fairness['score_parity']['score_gap']} ({region_fairness['score_parity']['relative_gap_pct']}%)")

        if not region_fairness['demographic_parity']['four_fifths_rule']:
            concerns.append("Region: Fails four-fifths rule - potential regional/ethnic bias")

        # ===== NATIONALITY ANALYSIS =====
        report.append("\n" + "=" * 70)
        report.append("## NATIONALITY ANALYSIS (Countries with n≥5)")
        report.append("=" * 70)

        nat_stats = self.analyze_by_attribute('country_name')
        nat_stats = nat_stats[nat_stats['count'] >= 5]
        report.append("\nStatistics by nationality:")
        report.append(nat_stats.to_string())

        # ===== INTERSECTIONAL ANALYSIS =====
        report.append("\n" + "=" * 70)
        report.append("## INTERSECTIONAL ANALYSIS (Gender × Region)")
        report.append("=" * 70)

        intersect_stats = self.intersectional_analysis('gender', 'region')
        report.append("\nMean scores by Gender × Region (n≥5):")
        report.append(intersect_stats.to_string())

        # ===== FINDINGS =====
        report.append("\n" + "=" * 70)
        report.append("## FINDINGS & RECOMMENDATIONS")
        report.append("=" * 70)

        if concerns:
            report.append("\n⚠️  POTENTIAL BIAS DETECTED:")
            for concern in concerns:
                report.append(f"  • {concern}")

            report.append("\n📋 RECOMMENDED MITIGATIONS:")
            report.append("  1. Remove/mask protected attributes from model input")
            report.append("  2. Apply score calibration to equalize group means")
            report.append("  3. Use adversarial debiasing during model training")
            report.append("  4. Implement threshold adjustments per group")
            report.append("  5. Regular bias audits with updated data")
            report.append("  6. Human review for edge cases near selection threshold")
        else:
            report.append("\n✓ No major bias patterns detected.")
            report.append("  Continue monitoring with regular audits.")

        report.append("\n" + "=" * 70)

        return "\n".join(report)


# --- Main Execution ---
if __name__ == "__main__":
    print("=" * 60)
    print("BIAS ANALYSIS WITH PROTECTED ATTRIBUTES")
    print("=" * 60)

    # Load data
    base_path = os.path.dirname(__file__)

    print("\nLoading datasets...")
    scores_df = pd.read_csv(os.path.join(base_path, "candidates_with_scores.csv"))
    protected_df = pd.read_csv(os.path.join(base_path, "protected_attributes.csv"))

    print(f"Candidates with scores: {len(scores_df)}")
    print(f"Protected attributes: {len(protected_df)}")

    # Initialize analyzer
    analyzer = ProtectedBiasAnalyzer(scores_df, protected_df)
    analyzer.overview()

    # Generate visualizations
    print("\nGenerating visualizations...")
    analyzer.generate_visualizations(base_path)

    # Generate report
    print("\nGenerating report...")
    report = analyzer.generate_report()
    print(report)

    # Save report
    report_path = os.path.join(base_path, "bias_report_protected.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to '{report_path}'")

    # Save merged data
    merged_path = os.path.join(base_path, "candidates_full_analysis.csv")
    analyzer.df.to_csv(merged_path, index=False)
    print(f"Full analysis data saved to '{merged_path}'")
