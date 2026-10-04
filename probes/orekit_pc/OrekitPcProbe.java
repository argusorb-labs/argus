import org.orekit.ssa.collision.shorttermencounter.probability.twod.Patera2005;
import org.hipparchus.analysis.integration.IterativeLegendreGaussIntegrator;

public final class OrekitPcProbe {
    public static void main(String[] args) {
        if (args.length != 5) throw new IllegalArgumentException("xm ym sigmaX sigmaY radius required");
        double[] p = new double[5];
        for (int i = 0; i < p.length; i++) {
            p[i] = Double.parseDouble(args[i]);
            if (!Double.isFinite(p[i])) throw new IllegalArgumentException("finite inputs required");
        }
        if (p[2] <= 0 || p[3] <= 0 || p[4] <= 0) throw new IllegalArgumentException("positive sigmas/radius required");
        double normal = new Patera2005().compute(p[0], p[1], p[2], p[3], p[4]).getValue();
        double tighter = new Patera2005(new IterativeLegendreGaussIntegrator(5, 1e-12, 1e-16), 200000)
                .compute(p[0], p[1], p[2], p[3], p[4]).getValue();
        if (!Double.isFinite(normal) || !Double.isFinite(tighter) || normal < 0 || normal > 1 || tighter < 0 || tighter > 1)
            throw new IllegalStateException("invalid probability");
        System.out.println("{\"method\":\"Patera2005\",\"default_pc\":" + normal + ",\"tighter_pc\":" + tighter + "}");
    }
}
